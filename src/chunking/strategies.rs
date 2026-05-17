//! Concrete chunking strategy implementations.
//!
//! Each strategy implements [`ChunkStrategy`] and controls how a [`Document`]
//! is split into overlapping [`Chunk`] fragments.
//!
//! | Strategy | Description |
//! |---|---|
//! | [`FixedSizeChunker`] | Splits on character boundaries with configurable overlap |
//! | [`SentenceChunker`] | Accumulates complete sentences until the size limit is reached |
//! | [`RecursiveChunker`] | Tries increasingly fine-grained separators recursively |
//! | [`MarkdownChunker`] | Respects Markdown heading and fenced-code-block boundaries |

use crate::chunking::chunk::Chunk;
use crate::chunking::config::ChunkConfig;
use crate::types::Document;

// ──────────────────────────────────────────────────────────────────────────────
// Public trait
// ──────────────────────────────────────────────────────────────────────────────

/// A strategy that splits a [`Document`] into a sequence of [`Chunk`]s.
pub trait ChunkStrategy: Send + Sync {
    /// Short, human-readable name for this strategy (e.g. `"fixed-size"`).
    fn name(&self) -> &'static str;

    /// Chunk the document according to the provided configuration.
    ///
    /// Implementations **must** respect `config.min_chunk_size`: any segment
    /// shorter than that threshold should be discarded rather than emitted.
    fn chunk(&self, doc: &Document, config: &ChunkConfig) -> Vec<Chunk>;
}

// ──────────────────────────────────────────────────────────────────────────────
// Helper utilities (module-private)
// ──────────────────────────────────────────────────────────────────────────────

/// Optionally strip whitespace from both ends of `s`, depending on config.
fn maybe_strip(s: &str, config: &ChunkConfig) -> String {
    if config.strip_whitespace {
        s.trim().to_string()
    } else {
        s.to_string()
    }
}

/// Build a `Chunk` from (`start_char`, `end_char`) window into `full_text`.
///
/// Returns `None` when the resulting content is shorter than
/// `config.min_chunk_size` (after optional whitespace stripping).
fn make_chunk(
    full_text: &str,
    start_char: usize,
    end_char: usize,
    chunk_index: usize,
    doc: &Document,
    config: &ChunkConfig,
) -> Option<Chunk> {
    // Collect char indices so we can safely slice.
    let chars: Vec<(usize, char)> = full_text.char_indices().collect();
    let byte_start = chars.get(start_char).map_or(full_text.len(), |(b, _)| *b);
    let byte_end = chars.get(end_char).map_or(full_text.len(), |(b, _)| *b);

    let raw = &full_text[byte_start..byte_end];
    let content = maybe_strip(raw, config);

    if content.chars().count() < config.min_chunk_size {
        return None;
    }

    Some(Chunk::new(
        content,
        doc.id.to_string(),
        chunk_index,
        start_char,
        end_char,
        doc.metadata.clone(),
    ))
}

/// Collect chunks by sliding a window of `chunk_size` chars over `text` with a
/// step of `config.step()` characters.  The window positions are expressed as
/// **character** offsets relative to `offset_in_doc` (the start of `text`
/// within the original document).
fn sliding_window_chunks(
    text: &str,
    offset_in_doc: usize,
    doc: &Document,
    config: &ChunkConfig,
    first_index: usize,
) -> Vec<Chunk> {
    let chars: Vec<char> = text.chars().collect();
    let total = chars.len();

    if total == 0 {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let mut chunk_index = first_index;
    let step = config.step();
    let mut start = 0usize;

    loop {
        let end = (start + config.chunk_size).min(total);

        // Build a sub-slice string for make_chunk.
        let sub: String = chars[start..end].iter().collect();
        let content = maybe_strip(&sub, config);

        if content.chars().count() >= config.min_chunk_size {
            let abs_start = offset_in_doc + start;
            let abs_end = offset_in_doc + end;
            chunks.push(Chunk::new(
                content,
                doc.id.to_string(),
                chunk_index,
                abs_start,
                abs_end,
                doc.metadata.clone(),
            ));
            chunk_index += 1;
        }

        if end == total {
            break;
        }
        start += step;
    }

    chunks
}

// ──────────────────────────────────────────────────────────────────────────────
// FixedSizeChunker
// ──────────────────────────────────────────────────────────────────────────────

/// Splits a document into fixed-size character windows with configurable overlap.
///
/// This is the simplest and fastest strategy.  It makes no attempt to
/// preserve linguistic boundaries; every chunk is exactly `chunk_size`
/// characters long (except possibly the final chunk which may be shorter).
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "chunking")]
/// # {
/// use oxirag::chunking::{ChunkConfig, ChunkStrategy, FixedSizeChunker};
/// use oxirag::types::Document;
///
/// let doc = Document::new("abcdefghij");
/// let config = ChunkConfig::default()
///     .with_chunk_size(4)
///     .with_chunk_overlap(1)
///     .with_min_chunk_size(1);
/// let chunks = FixedSizeChunker.chunk(&doc, &config);
/// assert!(!chunks.is_empty());
/// # }
/// ```
pub struct FixedSizeChunker;

impl ChunkStrategy for FixedSizeChunker {
    fn name(&self) -> &'static str {
        "fixed-size"
    }

    fn chunk(&self, doc: &Document, config: &ChunkConfig) -> Vec<Chunk> {
        sliding_window_chunks(&doc.content, 0, doc, config, 0)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// SentenceChunker
// ──────────────────────────────────────────────────────────────────────────────

/// Splits on sentence boundaries (`. `, `? `, `! `, `\n\n`) and accumulates
/// complete sentences until the chunk size limit is reached.
///
/// When an accumulated block exceeds `chunk_size`, it is emitted as a chunk and
/// the next block is seeded with the last sentence of the previous block
/// (providing overlap at sentence granularity).
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "chunking")]
/// # {
/// use oxirag::chunking::{ChunkConfig, ChunkStrategy, SentenceChunker};
/// use oxirag::types::Document;
///
/// let doc = Document::new("First sentence. Second sentence. Third sentence.");
/// let config = ChunkConfig::default().with_chunk_size(40).with_chunk_overlap(10);
/// let chunks = SentenceChunker.chunk(&doc, &config);
/// assert!(!chunks.is_empty());
/// # }
/// ```
pub struct SentenceChunker;

/// Terminators that end a sentence.  Order matters for the split logic.
const SENTENCE_TERMINATORS: &[&str] = &["\n\n", ". ", "? ", "! "];

impl SentenceChunker {
    /// Split `text` into a list of sentence-like segments, keeping the
    /// terminator attached to the preceding segment.
    fn split_sentences(text: &str) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }

        let mut sentences: Vec<String> = Vec::new();
        let mut remainder = text.to_string();

        loop {
            if remainder.is_empty() {
                break;
            }

            // Find the earliest terminator occurrence.
            let mut best_end: Option<usize> = None; // byte offset past the terminator

            for term in SENTENCE_TERMINATORS {
                if let Some(pos) = remainder.find(term) {
                    let end = pos + term.len();
                    if best_end.is_none() || end < best_end.unwrap_or(usize::MAX) {
                        best_end = Some(end);
                    }
                }
            }

            let Some(end) = best_end else {
                // No more terminators — everything left is one sentence.
                sentences.push(remainder.clone());
                break;
            };

            let sentence = remainder[..end].to_string();
            sentences.push(sentence);
            remainder = remainder[end..].to_string();
        }

        sentences
    }

    /// Compute the char-offset at which `segment` starts inside `full_text`,
    /// searching from `search_from` (char index).
    fn find_offset(full_text: &str, segment: &str, search_from: usize) -> usize {
        // Walk char-by-char to find the byte position of search_from, then
        // search the byte slice for the segment's bytes.
        let chars: Vec<(usize, char)> = full_text.char_indices().collect();
        let byte_from = chars.get(search_from).map_or(full_text.len(), |(b, _)| *b);
        let slice = &full_text[byte_from..];
        let byte_offset = slice.find(segment).unwrap_or(0);
        // Convert byte offset back to char offset.
        let prefix = &full_text[..byte_from + byte_offset];
        prefix.chars().count()
    }

    /// Build the overlap seed for the next chunk from `current_sentences`.
    ///
    /// Returns a list of sentences from the tail of `current_sentences` whose
    /// combined length is at least `chunk_overlap` (or all sentences if shorter).
    fn build_overlap(current_sentences: &[String], chunk_overlap: usize) -> Vec<String> {
        if chunk_overlap == 0 {
            return Vec::new();
        }
        let mut overlap_sentences: Vec<String> = Vec::new();
        let mut overlap_len = 0usize;
        for s in current_sentences.iter().rev() {
            let slen = s.chars().count();
            overlap_sentences.insert(0, s.clone());
            overlap_len += slen;
            if overlap_len >= chunk_overlap {
                break;
            }
        }
        overlap_sentences
    }
}

impl ChunkStrategy for SentenceChunker {
    fn name(&self) -> &'static str {
        "sentence"
    }

    fn chunk(&self, doc: &Document, config: &ChunkConfig) -> Vec<Chunk> {
        let sentences = Self::split_sentences(&doc.content);
        if sentences.is_empty() {
            return Vec::new();
        }

        let mut chunks: Vec<Chunk> = Vec::new();
        let mut current_sentences: Vec<String> = Vec::new();
        let mut current_len: usize = 0;
        let mut search_from_char: usize = 0;

        for sentence in &sentences {
            let sentence_len = sentence.chars().count();

            if current_len + sentence_len > config.chunk_size && !current_sentences.is_empty() {
                // Emit the current accumulation as a chunk.
                let text: String = current_sentences.join("");
                let content = maybe_strip(&text, config);

                if content.chars().count() >= config.min_chunk_size {
                    let start_char =
                        Self::find_offset(&doc.content, text.trim_start(), 0).min(search_from_char);
                    let end_char =
                        (start_char + content.chars().count()).min(doc.content.chars().count());
                    chunks.push(Chunk::new(
                        content,
                        doc.id.to_string(),
                        chunks.len(),
                        start_char,
                        end_char,
                        doc.metadata.clone(),
                    ));
                }

                // Seed next chunk with overlap sentences.
                let overlap = Self::build_overlap(&current_sentences, config.chunk_overlap);
                current_len = overlap.iter().map(|s| s.chars().count()).sum();
                current_sentences = overlap;
            }

            current_sentences.push(sentence.clone());
            current_len += sentence_len;
            search_from_char += sentence_len;
        }

        // Flush remaining sentences.
        if !current_sentences.is_empty() {
            let text: String = current_sentences.join("");
            let content = maybe_strip(&text, config);
            if content.chars().count() >= config.min_chunk_size {
                let total_chars = doc.content.chars().count();
                let end_char = total_chars;
                let start_char = end_char.saturating_sub(content.chars().count());
                chunks.push(Chunk::new(
                    content,
                    doc.id.to_string(),
                    chunks.len(),
                    start_char,
                    end_char,
                    doc.metadata.clone(),
                ));
            }
        }

        chunks
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// RecursiveChunker
// ──────────────────────────────────────────────────────────────────────────────

/// Recursively tries progressively finer separators until every segment fits
/// within `chunk_size`.
///
/// The separator hierarchy (tried in order) is:
///
/// 1. `"\n\n"` — paragraph break
/// 2. `"\n"` — newline
/// 3. `". "` — end of sentence (period + space)
/// 4. `"? "` — end of question
/// 5. `"! "` — end of exclamation
/// 6. `" "` — word boundary
/// 7. `""` — character-by-character (last resort)
///
/// Segments that are already within `chunk_size` are passed directly to the
/// sliding-window combiner; oversized segments are split recursively using the
/// next separator in the hierarchy.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "chunking")]
/// # {
/// use oxirag::chunking::{ChunkConfig, ChunkStrategy, RecursiveChunker};
/// use oxirag::types::Document;
///
/// let doc = Document::new("Paragraph one.\n\nParagraph two.\n\nParagraph three.");
/// let config = ChunkConfig::default().with_chunk_size(30).with_chunk_overlap(5);
/// let chunks = RecursiveChunker.chunk(&doc, &config);
/// assert!(!chunks.is_empty());
/// # }
/// ```
pub struct RecursiveChunker;

/// Separator hierarchy for [`RecursiveChunker`], from coarsest to finest.
const RECURSIVE_SEPARATORS: &[&str] = &["\n\n", "\n", ". ", "? ", "! ", " ", ""];

impl RecursiveChunker {
    /// Split `text` into pieces using the first separator in `seps` that
    /// produces more than one piece, or fall back to the next separator.
    ///
    /// Returns the final list of small-enough leaf segments paired with their
    /// byte-offsets within `text`.
    fn split_recursive<'a>(
        text: &'a str,
        seps: &[&str],
        chunk_size: usize,
    ) -> Vec<(usize, &'a str)> {
        // Base: text fits or no separators left.
        if text.chars().count() <= chunk_size || seps.is_empty() {
            return vec![(0, text)];
        }

        let sep = seps[0];
        let remaining = &seps[1..];

        // Special case: empty separator → split char by char.
        if sep.is_empty() {
            // Yield the text as a single leaf; sliding_window_chunks will
            // handle the fixed-size slicing.
            return vec![(0, text)];
        }

        let parts: Vec<&str> = text.split(sep).collect();

        if parts.len() <= 1 {
            // This separator doesn't help — try next.
            return Self::split_recursive(text, remaining, chunk_size);
        }

        // Accumulate the pieces back into segments of at most chunk_size.
        let mut results: Vec<(usize, &str)> = Vec::new();
        let mut byte_cursor: usize = 0;

        for part in &parts {
            let part_char_len = part.chars().count();

            if part_char_len > chunk_size {
                // Part itself is too large — recurse with a finer separator.
                let sub_results = Self::split_recursive(part, remaining, chunk_size);
                for (sub_off, sub_str) in sub_results {
                    results.push((byte_cursor + sub_off, sub_str));
                }
            } else {
                results.push((byte_cursor, part));
            }

            byte_cursor += part.len() + sep.len();
        }

        results
    }

    /// Compute character-offset boundaries for `leaves` within the original document.
    fn compute_boundaries(doc_content: &str, leaves: &[(usize, &str)]) -> Vec<(usize, usize)> {
        let mut boundaries: Vec<(usize, usize)> = Vec::new();
        let mut search_from = 0usize;
        for (_, part) in leaves {
            if part.is_empty() {
                boundaries.push((search_from, search_from));
                continue;
            }
            // Find the part in the original document starting from search_from.
            let chars_from: Vec<(usize, char)> =
                doc_content.char_indices().skip(search_from).collect();
            let byte_from = chars_from.first().map_or(doc_content.len(), |(b, _)| *b);
            let slice = &doc_content[byte_from..];
            let byte_pos = slice.find(part).unwrap_or(0);
            let abs_byte = byte_from + byte_pos;
            // Convert absolute byte to char count.
            let start_c = doc_content[..abs_byte].chars().count();
            let end_c = start_c + part.chars().count();
            boundaries.push((start_c, end_c));
            search_from = end_c;
        }
        boundaries
    }

    /// Compute the overlap tail of `current_text` as a new seed string and its
    /// start-char position.
    ///
    /// Returns `(seed_text, seed_start_char)`.  When overlap is disabled or the
    /// tail is empty, `part` and `part_start` are returned unchanged.
    fn compute_overlap_seed(
        current_text: &str,
        current_start_char: usize,
        part: &str,
        part_start: usize,
        config: &ChunkConfig,
    ) -> (String, usize) {
        if config.chunk_overlap > 0 {
            let ct_chars: Vec<char> = current_text.chars().collect();
            let overlap_start = ct_chars.len().saturating_sub(config.chunk_overlap);
            let overlap: String = ct_chars[overlap_start..].iter().collect();
            if !overlap.is_empty() {
                let total_before = current_start_char + current_text.chars().count();
                let overlap_char_len = overlap.chars().count();
                let overlap_start_char = total_before.saturating_sub(overlap_char_len);
                let new_text = format!("{overlap} {part}");
                return (new_text, overlap_start_char);
            }
        }
        (part.to_string(), part_start)
    }
}

impl ChunkStrategy for RecursiveChunker {
    fn name(&self) -> &'static str {
        "recursive"
    }

    fn chunk(&self, doc: &Document, config: &ChunkConfig) -> Vec<Chunk> {
        if doc.content.is_empty() {
            return Vec::new();
        }

        // Step 1: split into leaf segments using the separator hierarchy.
        let leaves = Self::split_recursive(&doc.content, RECURSIVE_SEPARATORS, config.chunk_size);

        let doc_char_count = doc.content.chars().count();
        let separator_boundaries = Self::compute_boundaries(&doc.content, &leaves);

        // Step 2: merge leaves into chunks, flushing when chunk_size is exceeded.
        let mut chunks: Vec<Chunk> = Vec::new();
        let mut current_text = String::new();
        let mut current_start_char: usize = 0;

        for (i, (_, part)) in leaves.iter().enumerate() {
            if part.is_empty() {
                continue;
            }

            let part_char_len = part.chars().count();
            let part_start = separator_boundaries[i].0;

            if !current_text.is_empty()
                && current_text.chars().count() + part_char_len > config.chunk_size
            {
                // Flush current accumulation.
                let content = maybe_strip(&current_text, config);
                if content.chars().count() >= config.min_chunk_size {
                    let end_char =
                        (current_start_char + current_text.chars().count()).min(doc_char_count);
                    chunks.push(Chunk::new(
                        content,
                        doc.id.to_string(),
                        chunks.len(),
                        current_start_char,
                        end_char,
                        doc.metadata.clone(),
                    ));
                }
                let (new_text, new_start) = Self::compute_overlap_seed(
                    &current_text,
                    current_start_char,
                    part,
                    part_start,
                    config,
                );
                current_text = new_text;
                current_start_char = new_start;
            } else if current_text.is_empty() {
                current_start_char = part_start;
                current_text = part.to_string();
            } else {
                current_text.push(' ');
                current_text.push_str(part);
            }
        }

        // Flush remainder.
        if !current_text.is_empty() {
            let content = maybe_strip(&current_text, config);
            if content.chars().count() >= config.min_chunk_size {
                let end_char =
                    (current_start_char + current_text.chars().count()).min(doc_char_count);
                chunks.push(Chunk::new(
                    content,
                    doc.id.to_string(),
                    chunks.len(),
                    current_start_char,
                    end_char,
                    doc.metadata.clone(),
                ));
            }
        }

        // If no chunks were emitted (e.g. very short doc), fall back to a
        // single fixed-size pass.
        if chunks.is_empty() {
            let content = maybe_strip(&doc.content, config);
            if content.chars().count() >= config.min_chunk_size {
                chunks.push(Chunk::new(
                    content,
                    doc.id.to_string(),
                    0,
                    0,
                    doc.content.chars().count(),
                    doc.metadata.clone(),
                ));
            }
        }

        chunks
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// MarkdownChunker
// ──────────────────────────────────────────────────────────────────────────────

/// Splits a Markdown document at heading boundaries (`# `, `## `, `### `).
///
/// Each section (the heading line plus all body text until the next heading of
/// the same or higher level) becomes a candidate segment.  If a section exceeds
/// `chunk_size`, it is further split by sliding window exactly like
/// [`FixedSizeChunker`].  Fenced code blocks (` ``` ` … ` ``` `) are preserved
/// intact and never split mid-block.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "chunking")]
/// # {
/// use oxirag::chunking::{ChunkConfig, ChunkStrategy, MarkdownChunker};
/// use oxirag::types::Document;
///
/// let md = "# Introduction\n\nHello world.\n\n## Details\n\nMore text here.";
/// let doc = Document::new(md);
/// let config = ChunkConfig::default().with_chunk_size(80).with_chunk_overlap(10);
/// let chunks = MarkdownChunker.chunk(&doc, &config);
/// assert!(chunks.len() >= 2);
/// # }
/// ```
pub struct MarkdownChunker;

/// Returns true when `line` starts a Markdown heading (ATX style, up to `###`).
fn is_heading(line: &str) -> bool {
    let trimmed = line.trim_start_matches('#');
    let hashes = line.len() - trimmed.len();
    (1..=3).contains(&hashes) && trimmed.starts_with(' ')
}

/// Returns true when `line` is a fenced code-block delimiter (` ``` ` or `~~~`).
fn is_fence(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.starts_with("```") || trimmed.starts_with("~~~")
}

impl MarkdownChunker {
    /// Extract text segments by splitting on heading boundaries.
    ///
    /// Lines inside fenced code blocks are never used as split points.
    fn extract_sections(text: &str) -> Vec<String> {
        let mut sections: Vec<String> = Vec::new();
        let mut current: Vec<&str> = Vec::new();
        let mut in_fence = false;

        for line in text.lines() {
            if is_fence(line) {
                in_fence = !in_fence;
                current.push(line);
                continue;
            }

            if !in_fence && is_heading(line) && !current.is_empty() {
                sections.push(current.join("\n"));
                current.clear();
            }

            current.push(line);
        }

        if !current.is_empty() {
            sections.push(current.join("\n"));
        }

        sections
    }

    /// Locate the start character of `section` in `doc_content`, searching
    /// from `search_from_char`.
    fn section_start_char(doc_content: &str, section: &str, search_from_char: usize) -> usize {
        let chars_from: Vec<(usize, char)> =
            doc_content.char_indices().skip(search_from_char).collect();
        let byte_from = chars_from.first().map_or(doc_content.len(), |(b, _)| *b);
        let slice = &doc_content[byte_from..];
        let trimmed_section = section.trim_start();
        let byte_pos = slice.find(trimmed_section).unwrap_or(0);
        let abs_byte = byte_from + byte_pos;
        doc_content[..abs_byte].chars().count()
    }
}

impl ChunkStrategy for MarkdownChunker {
    fn name(&self) -> &'static str {
        "markdown"
    }

    fn chunk(&self, doc: &Document, config: &ChunkConfig) -> Vec<Chunk> {
        if doc.content.is_empty() {
            return Vec::new();
        }

        let sections = Self::extract_sections(&doc.content);

        // For each section, build chunks.  Large sections get sliding-window
        // treatment; small ones become a single chunk.
        let mut all_chunks: Vec<Chunk> = Vec::new();
        let mut search_from_char: usize = 0;
        let doc_char_count = doc.content.chars().count();

        for section in &sections {
            if section.trim().is_empty() {
                search_from_char += section.chars().count() + 1; // +1 for the \n separator
                continue;
            }

            let start_char = Self::section_start_char(&doc.content, section, search_from_char);
            let section_char_len = section.chars().count();

            if section_char_len <= config.chunk_size {
                let content = maybe_strip(section, config);
                if content.chars().count() >= config.min_chunk_size {
                    let end_char = (start_char + section_char_len).min(doc_char_count);
                    all_chunks.push(Chunk::new(
                        content,
                        doc.id.to_string(),
                        all_chunks.len(),
                        start_char,
                        end_char,
                        doc.metadata.clone(),
                    ));
                }
            } else {
                // Section too large — apply sliding window within this section.
                let sub_chunks =
                    sliding_window_chunks(section, start_char, doc, config, all_chunks.len());
                // Re-index chunks sequentially.
                for mut sub in sub_chunks {
                    sub.chunk_index = all_chunks.len();
                    sub.metadata
                        .insert("chunk_index".to_string(), sub.chunk_index.to_string());
                    all_chunks.push(sub);
                }
            }

            search_from_char = search_from_char
                .saturating_add(section_char_len)
                .saturating_add(1);
        }

        // If the document had no headings at all, fall back to fixed-size.
        if all_chunks.is_empty() {
            return sliding_window_chunks(&doc.content, 0, doc, config, 0);
        }

        all_chunks
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Helper re-export for chunker.rs
// ──────────────────────────────────────────────────────────────────────────────

/// Build a `Chunk` from absolute character positions in `doc.content`.
///
/// Used internally when strategies have already produced position data.
#[allow(dead_code)]
pub(crate) fn chunk_from_positions(
    doc: &Document,
    start_char: usize,
    end_char: usize,
    chunk_index: usize,
    config: &ChunkConfig,
) -> Option<Chunk> {
    make_chunk(&doc.content, start_char, end_char, chunk_index, doc, config)
}
