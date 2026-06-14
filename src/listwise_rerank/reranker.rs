//! `RankGPT` sliding-window listwise reranker.
use crate::listwise_rerank::judge::LexicalListwiseJudge;
use crate::listwise_rerank::types::{ListwiseError, ListwiseJudge, ListwiseResult, WindowConfig};
use crate::types::{Document, SearchResult};

// ── ListwiseReranker ──────────────────────────────────────────────────────────

/// Reranks retrieval results with the `RankGPT` sliding-window strategy.
///
/// A window of [`window_size`](WindowConfig::window_size) candidates slides from
/// the back of the ranking toward the front in steps of
/// [`step`](WindowConfig::step). Each window is permuted best→worst by the
/// [`ListwiseJudge`], and the permuted order is written back. Repeated passes
/// bubble the most relevant items toward the top, exactly as described by Sun
/// et al. (2023).
#[derive(Debug, Clone)]
pub struct ListwiseReranker<J: ListwiseJudge> {
    /// Configuration controlling the sliding window.
    pub config: WindowConfig,
    /// The judge used to permute each window.
    pub judge: J,
}

impl ListwiseReranker<LexicalListwiseJudge> {
    /// Create a reranker backed by the default [`LexicalListwiseJudge`].
    #[must_use]
    pub fn new(config: WindowConfig) -> Self {
        Self {
            config,
            judge: LexicalListwiseJudge::new(),
        }
    }
}

impl<J: ListwiseJudge> ListwiseReranker<J> {
    /// Create a reranker with a custom judge.
    #[must_use]
    pub fn with_judge(config: WindowConfig, judge: J) -> Self {
        Self { config, judge }
    }

    /// Run the sliding-window permutation passes and return the final ordering.
    ///
    /// The returned vector contains indices into `candidates`, in best→worst
    /// order.
    fn slide(&self, query: &str, candidates: &[Document]) -> Vec<usize> {
        let n = candidates.len();
        let mut order: Vec<usize> = (0..n).collect();
        // `n >= 1` (callers guard against empty input) and `effective_window`
        // is already at least 1, so clamping to `n` keeps `window` in `1..=n`.
        let window = self.config.effective_window().min(n);
        let step = self.config.effective_step();

        let mut end = n;
        while end > 0 {
            let start = end.saturating_sub(window);
            let window_slice: Vec<usize> = order[start..end].to_vec();
            let window_docs: Vec<Document> = window_slice
                .iter()
                .map(|&i| candidates[i].clone())
                .collect();
            let perm = self.judge.permute(query, &window_docs);
            let reordered: Vec<usize> = perm.iter().map(|&p| window_slice[p]).collect();
            order[start..end].copy_from_slice(&reordered);
            if start == 0 {
                break;
            }
            end -= step;
        }
        order
    }

    /// Rerank `results` for `query`, producing [`ListwiseResult`]s in new order.
    ///
    /// # Errors
    ///
    /// Returns [`ListwiseError::EmptyQuery`] if `query` is empty (after trimming).
    /// Returns [`ListwiseError::EmptyCandidates`] if `results` is empty.
    pub fn rerank(
        &self,
        query: &str,
        results: &[SearchResult],
    ) -> Result<Vec<ListwiseResult>, ListwiseError> {
        if query.trim().is_empty() {
            return Err(ListwiseError::EmptyQuery);
        }
        if results.is_empty() {
            return Err(ListwiseError::EmptyCandidates);
        }

        let candidates: Vec<Document> = results.iter().map(|r| r.document.clone()).collect();
        let order = self.slide(query, &candidates);

        let mut reranked: Vec<ListwiseResult> = order
            .into_iter()
            .enumerate()
            .map(|(new_rank, orig_idx)| ListwiseResult {
                document: results[orig_idx].document.clone(),
                original_score: results[orig_idx].score,
                original_rank: orig_idx,
                new_rank,
            })
            .collect();

        if self.config.top_n > 0 {
            reranked.truncate(self.config.top_n);
        }
        Ok(reranked)
    }

    /// Rerank `results` and return fresh [`SearchResult`]s carrying the new ranks.
    ///
    /// Each returned [`SearchResult`] keeps its original score but receives a new
    /// 0-indexed [`rank`](SearchResult::rank) reflecting the listwise order.
    ///
    /// # Errors
    ///
    /// Propagates [`ListwiseError`] from [`rerank`](Self::rerank).
    pub fn rerank_to_results(
        &self,
        query: &str,
        results: &[SearchResult],
    ) -> Result<Vec<SearchResult>, ListwiseError> {
        let reranked = self.rerank(query, results)?;
        Ok(reranked
            .into_iter()
            .map(|r| SearchResult {
                document: r.document,
                score: r.original_score,
                rank: r.new_rank,
            })
            .collect())
    }
}

impl Default for ListwiseReranker<LexicalListwiseJudge> {
    fn default() -> Self {
        Self::new(WindowConfig::default())
    }
}
