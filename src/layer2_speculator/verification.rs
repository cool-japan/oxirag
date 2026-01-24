//! Multi-stage verification pipeline for the Speculator layer.
//!
//! This module provides a flexible pipeline for verifying draft answers
//! through multiple stages, with optional early termination based on
//! confidence thresholds.

use crate::error::SpeculatorError;
use crate::layer2_speculator::calibration::ConfidenceCalibrator;
use crate::types::{Draft, SearchResult, SpeculationDecision, SpeculationResult};
use std::collections::HashSet;

/// Result from a single verification stage.
#[derive(Debug, Clone)]
pub struct StageResult {
    /// Name of the stage that produced this result.
    pub stage_name: String,
    /// Confidence score from this stage (0.0 to 1.0).
    pub confidence: f32,
    /// Whether verification should continue to the next stage.
    pub should_continue: bool,
    /// Optional notes or explanation from this stage.
    pub notes: Option<String>,
}

impl StageResult {
    /// Create a new stage result.
    #[must_use]
    pub fn new(stage_name: impl Into<String>, confidence: f32) -> Self {
        Self {
            stage_name: stage_name.into(),
            confidence: confidence.clamp(0.0, 1.0),
            should_continue: true,
            notes: None,
        }
    }

    /// Set whether verification should continue.
    #[must_use]
    pub fn with_continue(mut self, should_continue: bool) -> Self {
        self.should_continue = should_continue;
        self
    }

    /// Add notes to this stage result.
    #[must_use]
    pub fn with_notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = Some(notes.into());
        self
    }
}

/// Trait for verification stages in the pipeline.
pub trait VerificationStage: Send + Sync {
    /// Get the name of this verification stage.
    fn name(&self) -> &str;

    /// Verify the draft against the context.
    ///
    /// # Arguments
    /// * `draft` - The draft answer to verify
    /// * `context` - The retrieved documents providing context
    ///
    /// # Returns
    /// A stage result with confidence and continuation decision.
    fn verify(&self, draft: &Draft, context: &[SearchResult]) -> StageResult;
}

/// Multi-stage verification pipeline.
pub struct VerificationPipeline {
    stages: Vec<Box<dyn VerificationStage>>,
    calibrator: Option<ConfidenceCalibrator>,
    early_accept_threshold: f32,
    early_reject_threshold: f32,
    aggregation_method: AggregationMethod,
}

/// Method for aggregating confidence scores across stages.
#[derive(Debug, Clone, Copy, Default)]
pub enum AggregationMethod {
    /// Use the average of all stage confidences.
    #[default]
    Average,
    /// Use the minimum confidence (most conservative).
    Minimum,
    /// Use the maximum confidence (most optimistic).
    Maximum,
    /// Use a weighted average based on stage order.
    WeightedAverage,
    /// Use the last stage's confidence.
    LastStage,
}

impl VerificationPipeline {
    /// Create a new empty verification pipeline.
    #[must_use]
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            calibrator: None,
            early_accept_threshold: 0.95,
            early_reject_threshold: 0.1,
            aggregation_method: AggregationMethod::Average,
        }
    }

    /// Add a verification stage to the pipeline.
    #[must_use]
    pub fn add_stage(mut self, stage: Box<dyn VerificationStage>) -> Self {
        self.stages.push(stage);
        self
    }

    /// Set a confidence calibrator.
    #[must_use]
    pub fn with_calibrator(mut self, calibrator: ConfidenceCalibrator) -> Self {
        self.calibrator = Some(calibrator);
        self
    }

    /// Set the early accept threshold.
    #[must_use]
    pub fn with_early_accept_threshold(mut self, threshold: f32) -> Self {
        self.early_accept_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the early reject threshold.
    #[must_use]
    pub fn with_early_reject_threshold(mut self, threshold: f32) -> Self {
        self.early_reject_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the aggregation method.
    #[must_use]
    pub fn with_aggregation_method(mut self, method: AggregationMethod) -> Self {
        self.aggregation_method = method;
        self
    }

    /// Verify a draft through the pipeline.
    ///
    /// # Errors
    ///
    /// This function does not currently return errors, but the signature
    /// allows for future extensions where verification stages may fail.
    #[allow(clippy::unused_async)]
    pub async fn verify(
        &self,
        draft: &Draft,
        context: &[SearchResult],
    ) -> Result<SpeculationResult, SpeculatorError> {
        if self.stages.is_empty() {
            // No stages, return neutral result
            return Ok(SpeculationResult::new(SpeculationDecision::Revise, 0.5)
                .with_explanation("No verification stages configured"));
        }

        let mut stage_results = Vec::new();
        let mut all_notes = Vec::new();

        for stage in &self.stages {
            let result = stage.verify(draft, context);

            // Collect notes
            if let Some(ref notes) = result.notes {
                all_notes.push(format!("[{}] {}", result.stage_name, notes));
            }

            stage_results.push(result.clone());

            // Check for early termination
            if !result.should_continue {
                break;
            }

            // Early accept/reject based on thresholds
            if result.confidence >= self.early_accept_threshold {
                break;
            }
            if result.confidence <= self.early_reject_threshold {
                break;
            }
        }

        // Aggregate confidence scores
        let raw_confidence = self.aggregate_confidence(&stage_results);

        // Apply calibration if available
        let confidence = self
            .calibrator
            .as_ref()
            .map_or(raw_confidence, |c| c.calibrate(raw_confidence));

        // Determine decision
        let decision = if confidence >= self.early_accept_threshold {
            SpeculationDecision::Accept
        } else if confidence <= self.early_reject_threshold {
            SpeculationDecision::Reject
        } else {
            SpeculationDecision::Revise
        };

        // Build explanation
        let explanation = if all_notes.is_empty() {
            format!(
                "Verification completed through {} stage(s) with confidence {:.2}",
                stage_results.len(),
                confidence
            )
        } else {
            all_notes.join("; ")
        };

        Ok(SpeculationResult::new(decision, confidence).with_explanation(explanation))
    }

    /// Aggregate confidence scores based on the configured method.
    #[allow(clippy::cast_precision_loss)]
    fn aggregate_confidence(&self, results: &[StageResult]) -> f32 {
        if results.is_empty() {
            return 0.5;
        }

        match self.aggregation_method {
            AggregationMethod::Average => {
                let sum: f32 = results.iter().map(|r| r.confidence).sum();
                sum / results.len() as f32
            }
            AggregationMethod::Minimum => results
                .iter()
                .map(|r| r.confidence)
                .fold(f32::MAX, f32::min),
            AggregationMethod::Maximum => results
                .iter()
                .map(|r| r.confidence)
                .fold(f32::MIN, f32::max),
            AggregationMethod::WeightedAverage => {
                let weighted_sum: f32 = results
                    .iter()
                    .enumerate()
                    .map(|(i, r)| r.confidence * (i + 1) as f32)
                    .sum();
                let weight_total: f32 = (1..=results.len()).map(|i| i as f32).sum();
                weighted_sum / weight_total
            }
            AggregationMethod::LastStage => results.last().map_or(0.5, |r| r.confidence),
        }
    }

    /// Get the number of stages in the pipeline.
    #[must_use]
    pub fn num_stages(&self) -> usize {
        self.stages.len()
    }

    /// Get the early accept threshold.
    #[must_use]
    pub fn early_accept_threshold(&self) -> f32 {
        self.early_accept_threshold
    }

    /// Get the early reject threshold.
    #[must_use]
    pub fn early_reject_threshold(&self) -> f32 {
        self.early_reject_threshold
    }
}

impl Default for VerificationPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// Built-in verification stages

/// Verification stage that checks for keyword overlap.
pub struct KeywordMatchStage {
    name: String,
    min_overlap_ratio: f32,
}

impl KeywordMatchStage {
    /// Create a new keyword match stage.
    #[must_use]
    pub fn new() -> Self {
        Self {
            name: "keyword_match".to_string(),
            min_overlap_ratio: 0.1,
        }
    }

    /// Set the minimum overlap ratio for passing.
    #[must_use]
    pub fn with_min_overlap(mut self, ratio: f32) -> Self {
        self.min_overlap_ratio = ratio.clamp(0.0, 1.0);
        self
    }
}

impl Default for KeywordMatchStage {
    fn default() -> Self {
        Self::new()
    }
}

impl VerificationStage for KeywordMatchStage {
    fn name(&self) -> &str {
        &self.name
    }

    fn verify(&self, draft: &Draft, context: &[SearchResult]) -> StageResult {
        if context.is_empty() {
            return StageResult::new(&self.name, 0.5)
                .with_notes("No context provided for keyword matching");
        }

        // Extract keywords from context
        let context_text: String = context
            .iter()
            .map(|r| r.document.content.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        let context_words: HashSet<String> = context_text
            .to_lowercase()
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .map(String::from)
            .collect();

        let draft_words: HashSet<String> = draft
            .content
            .to_lowercase()
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .map(String::from)
            .collect();

        if draft_words.is_empty() {
            return StageResult::new(&self.name, 0.0)
                .with_notes("Draft contains no significant keywords")
                .with_continue(true);
        }

        #[allow(clippy::cast_precision_loss)]
        let overlap_count = draft_words.intersection(&context_words).count();
        #[allow(clippy::cast_precision_loss)]
        let overlap_ratio = overlap_count as f32 / draft_words.len() as f32;

        let confidence = overlap_ratio.clamp(0.0, 1.0);

        let notes = format!(
            "Found {}/{} draft keywords in context ({:.1}%)",
            overlap_count,
            draft_words.len(),
            overlap_ratio * 100.0
        );

        StageResult::new(&self.name, confidence)
            .with_notes(notes)
            .with_continue(confidence >= self.min_overlap_ratio)
    }
}

/// Verification stage that checks for semantic similarity.
pub struct SemanticSimilarityStage {
    name: String,
    min_similarity: f32,
}

impl SemanticSimilarityStage {
    /// Create a new semantic similarity stage.
    #[must_use]
    pub fn new() -> Self {
        Self {
            name: "semantic_similarity".to_string(),
            min_similarity: 0.5,
        }
    }

    /// Set the minimum similarity threshold.
    #[must_use]
    pub fn with_min_similarity(mut self, similarity: f32) -> Self {
        self.min_similarity = similarity.clamp(0.0, 1.0);
        self
    }

    /// Compute simple character n-gram based similarity.
    fn ngram_similarity(text1: &str, text2: &str, n: usize) -> f32 {
        let ngrams1: HashSet<String> = Self::extract_ngrams(text1, n);
        let ngrams2: HashSet<String> = Self::extract_ngrams(text2, n);

        if ngrams1.is_empty() || ngrams2.is_empty() {
            return 0.0;
        }

        #[allow(clippy::cast_precision_loss)]
        let intersection = ngrams1.intersection(&ngrams2).count() as f32;
        #[allow(clippy::cast_precision_loss)]
        let union = ngrams1.union(&ngrams2).count() as f32;

        intersection / union
    }

    /// Extract character n-grams from text.
    fn extract_ngrams(text: &str, n: usize) -> HashSet<String> {
        let chars: Vec<char> = text.to_lowercase().chars().collect();
        if chars.len() < n {
            return HashSet::new();
        }

        chars.windows(n).map(|w| w.iter().collect()).collect()
    }
}

impl Default for SemanticSimilarityStage {
    fn default() -> Self {
        Self::new()
    }
}

impl VerificationStage for SemanticSimilarityStage {
    fn name(&self) -> &str {
        &self.name
    }

    fn verify(&self, draft: &Draft, context: &[SearchResult]) -> StageResult {
        if context.is_empty() {
            return StageResult::new(&self.name, 0.5).with_notes("No context for similarity check");
        }

        // Combine context texts
        let context_text: String = context
            .iter()
            .map(|r| r.document.content.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        // Compute n-gram similarities at different levels
        let sim_3gram = Self::ngram_similarity(&draft.content, &context_text, 3);
        let sim_4gram = Self::ngram_similarity(&draft.content, &context_text, 4);
        let sim_5gram = Self::ngram_similarity(&draft.content, &context_text, 5);

        // Weight higher n-grams more heavily (they capture phrases better)
        let similarity = sim_3gram * 0.2 + sim_4gram * 0.3 + sim_5gram * 0.5;
        let confidence = similarity.clamp(0.0, 1.0);

        let notes = format!(
            "N-gram similarity: 3-gram={sim_3gram:.2}, 4-gram={sim_4gram:.2}, 5-gram={sim_5gram:.2}"
        );

        StageResult::new(&self.name, confidence)
            .with_notes(notes)
            .with_continue(confidence >= self.min_similarity)
    }
}

/// Verification stage that checks for factual consistency.
pub struct FactualConsistencyStage {
    name: String,
    negation_patterns: Vec<String>,
    contradiction_penalty: f32,
}

impl FactualConsistencyStage {
    /// Create a new factual consistency stage.
    #[must_use]
    pub fn new() -> Self {
        Self {
            name: "factual_consistency".to_string(),
            negation_patterns: vec![
                "not".to_string(),
                "never".to_string(),
                "no".to_string(),
                "none".to_string(),
                "neither".to_string(),
                "without".to_string(),
                "isn't".to_string(),
                "aren't".to_string(),
                "wasn't".to_string(),
                "weren't".to_string(),
                "don't".to_string(),
                "doesn't".to_string(),
                "didn't".to_string(),
                "won't".to_string(),
                "wouldn't".to_string(),
                "can't".to_string(),
                "cannot".to_string(),
                "couldn't".to_string(),
            ],
            contradiction_penalty: 0.2,
        }
    }

    /// Set the penalty for contradictions.
    #[must_use]
    pub fn with_contradiction_penalty(mut self, penalty: f32) -> Self {
        self.contradiction_penalty = penalty.clamp(0.0, 1.0);
        self
    }

    /// Check if a sentence contains negation.
    fn contains_negation(&self, text: &str) -> bool {
        let lower = text.to_lowercase();
        self.negation_patterns.iter().any(|p| lower.contains(p))
    }

    /// Extract simple assertions from text.
    fn extract_assertions(text: &str) -> Vec<(String, String)> {
        // Very simple subject-predicate extraction
        let mut assertions = Vec::new();

        for sentence in text.split(['.', '!', '?']) {
            let sentence = sentence.trim().to_lowercase();
            if sentence.len() < 5 {
                continue;
            }

            // Split on "is", "are", "was", "were" as simple predicate markers
            for marker in ["is", "are", "was", "were"] {
                if let Some(pos) = sentence.find(&format!(" {marker} ")) {
                    let subject = sentence[..pos].trim().to_string();
                    let predicate = sentence[pos + marker.len() + 2..].trim().to_string();
                    if !subject.is_empty() && !predicate.is_empty() {
                        assertions.push((subject, predicate));
                    }
                }
            }
        }

        assertions
    }
}

impl Default for FactualConsistencyStage {
    fn default() -> Self {
        Self::new()
    }
}

impl VerificationStage for FactualConsistencyStage {
    fn name(&self) -> &str {
        &self.name
    }

    fn verify(&self, draft: &Draft, context: &[SearchResult]) -> StageResult {
        if context.is_empty() {
            return StageResult::new(&self.name, 0.5)
                .with_notes("No context for consistency check");
        }

        let context_text: String = context
            .iter()
            .map(|r| r.document.content.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        let draft_assertions = Self::extract_assertions(&draft.content);
        let context_assertions = Self::extract_assertions(&context_text);

        if draft_assertions.is_empty() {
            return StageResult::new(&self.name, 0.7)
                .with_notes("No clear assertions found in draft");
        }

        let mut contradictions = 0;
        let mut supported = 0;

        for (draft_subj, draft_pred) in &draft_assertions {
            let draft_negated = self.contains_negation(draft_pred);

            for (ctx_subj, ctx_pred) in &context_assertions {
                // Check if subjects overlap
                if draft_subj.contains(ctx_subj) || ctx_subj.contains(draft_subj) {
                    let ctx_negated = self.contains_negation(ctx_pred);

                    // Check for predicate overlap with opposite negation
                    if draft_negated != ctx_negated {
                        // Potential contradiction
                        if draft_pred.split_whitespace().any(|w| ctx_pred.contains(w)) {
                            contradictions += 1;
                        }
                    } else if draft_pred.split_whitespace().any(|w| ctx_pred.contains(w)) {
                        // Same polarity with predicate overlap = support
                        supported += 1;
                    }
                }
            }
        }

        #[allow(clippy::cast_precision_loss)]
        let base_confidence = if supported > 0 || contradictions > 0 {
            supported as f32 / (supported + contradictions) as f32
        } else {
            0.6 // Neutral when no matches found
        };

        #[allow(clippy::cast_precision_loss)]
        let penalty = contradictions as f32 * self.contradiction_penalty;
        let confidence = (base_confidence - penalty).clamp(0.0, 1.0);

        let notes = format!(
            "Found {} supported and {} potential contradictions in {} assertions",
            supported,
            contradictions,
            draft_assertions.len()
        );

        StageResult::new(&self.name, confidence).with_notes(notes)
    }
}

/// Builder for creating verification pipelines with common configurations.
pub struct PipelineBuilder {
    pipeline: VerificationPipeline,
}

impl PipelineBuilder {
    /// Create a new pipeline builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            pipeline: VerificationPipeline::new(),
        }
    }

    /// Add the keyword match stage.
    #[must_use]
    pub fn with_keyword_match(mut self) -> Self {
        self.pipeline = self.pipeline.add_stage(Box::new(KeywordMatchStage::new()));
        self
    }

    /// Add the semantic similarity stage.
    #[must_use]
    pub fn with_semantic_similarity(mut self) -> Self {
        self.pipeline = self
            .pipeline
            .add_stage(Box::new(SemanticSimilarityStage::new()));
        self
    }

    /// Add the factual consistency stage.
    #[must_use]
    pub fn with_factual_consistency(mut self) -> Self {
        self.pipeline = self
            .pipeline
            .add_stage(Box::new(FactualConsistencyStage::new()));
        self
    }

    /// Add all standard stages.
    #[must_use]
    pub fn with_standard_stages(self) -> Self {
        self.with_keyword_match()
            .with_semantic_similarity()
            .with_factual_consistency()
    }

    /// Set the early accept threshold.
    #[must_use]
    pub fn early_accept(mut self, threshold: f32) -> Self {
        self.pipeline = self.pipeline.with_early_accept_threshold(threshold);
        self
    }

    /// Set the early reject threshold.
    #[must_use]
    pub fn early_reject(mut self, threshold: f32) -> Self {
        self.pipeline = self.pipeline.with_early_reject_threshold(threshold);
        self
    }

    /// Set the aggregation method.
    #[must_use]
    pub fn aggregation(mut self, method: AggregationMethod) -> Self {
        self.pipeline = self.pipeline.with_aggregation_method(method);
        self
    }

    /// Set a calibrator.
    #[must_use]
    pub fn calibrator(mut self, calibrator: ConfidenceCalibrator) -> Self {
        self.pipeline = self.pipeline.with_calibrator(calibrator);
        self
    }

    /// Build the pipeline.
    #[must_use]
    pub fn build(self) -> VerificationPipeline {
        self.pipeline
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::unnecessary_to_owned)]
mod tests {
    use super::*;
    use crate::types::Document;

    fn create_context() -> Vec<SearchResult> {
        vec![
            SearchResult::new(
                Document::new("Paris is the capital of France. It is known for the Eiffel Tower."),
                0.9,
                0,
            ),
            SearchResult::new(
                Document::new(
                    "France is a country in Western Europe with a population of about 67 million.",
                ),
                0.85,
                1,
            ),
        ]
    }

    #[test]
    fn test_stage_result_creation() {
        let result = StageResult::new("test_stage", 0.8)
            .with_continue(true)
            .with_notes("Test notes");

        assert_eq!(result.stage_name, "test_stage");
        assert_eq!(result.confidence, 0.8);
        assert!(result.should_continue);
        assert_eq!(result.notes, Some("Test notes".to_string()));
    }

    #[test]
    fn test_stage_result_clamping() {
        let result = StageResult::new("test", 1.5);
        assert_eq!(result.confidence, 1.0);

        let result2 = StageResult::new("test", -0.5);
        assert_eq!(result2.confidence, 0.0);
    }

    #[test]
    fn test_pipeline_creation() {
        let pipeline = VerificationPipeline::new()
            .with_early_accept_threshold(0.9)
            .with_early_reject_threshold(0.2);

        assert_eq!(pipeline.num_stages(), 0);
        assert_eq!(pipeline.early_accept_threshold(), 0.9);
        assert_eq!(pipeline.early_reject_threshold(), 0.2);
    }

    #[test]
    fn test_pipeline_add_stage() {
        let pipeline = VerificationPipeline::new().add_stage(Box::new(KeywordMatchStage::new()));

        assert_eq!(pipeline.num_stages(), 1);
    }

    #[tokio::test]
    async fn test_pipeline_empty() {
        let pipeline = VerificationPipeline::new();
        let draft = Draft::new("Test answer", "Test question");
        let context = create_context();

        let result = pipeline.verify(&draft, &context).await.unwrap();

        assert!(matches!(result.decision, SpeculationDecision::Revise));
        assert_eq!(result.confidence, 0.5);
    }

    #[tokio::test]
    async fn test_pipeline_with_stages() {
        let pipeline = VerificationPipeline::new()
            .add_stage(Box::new(KeywordMatchStage::new()))
            .add_stage(Box::new(SemanticSimilarityStage::new()));

        let draft = Draft::new(
            "Paris is the capital of France and is known for the Eiffel Tower.",
            "What is the capital of France?",
        );
        let context = create_context();

        let result = pipeline.verify(&draft, &context).await.unwrap();

        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_keyword_match_stage() {
        let stage = KeywordMatchStage::new();
        let draft = Draft::new(
            "Paris is the capital of France with the Eiffel Tower.",
            "test",
        );
        let context = create_context();

        let result = stage.verify(&draft, &context);

        assert_eq!(result.stage_name, "keyword_match");
        assert!(result.confidence > 0.0);
        assert!(result.notes.is_some());
    }

    #[test]
    fn test_keyword_match_no_context() {
        let stage = KeywordMatchStage::new();
        let draft = Draft::new("Some text", "test");

        let result = stage.verify(&draft, &[]);

        assert_eq!(result.confidence, 0.5);
    }

    #[test]
    fn test_keyword_match_empty_draft() {
        let stage = KeywordMatchStage::new();
        let draft = Draft::new("a", "test"); // No words > 3 chars
        let context = create_context();

        let result = stage.verify(&draft, &context);

        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn test_semantic_similarity_stage() {
        let stage = SemanticSimilarityStage::new();
        let draft = Draft::new(
            "Paris is the capital of France known for the Eiffel Tower.",
            "test",
        );
        let context = create_context();

        let result = stage.verify(&draft, &context);

        assert_eq!(result.stage_name, "semantic_similarity");
        assert!(result.confidence >= 0.0);
    }

    #[test]
    fn test_semantic_similarity_no_context() {
        let stage = SemanticSimilarityStage::new();
        let draft = Draft::new("Some text", "test");

        let result = stage.verify(&draft, &[]);

        assert_eq!(result.confidence, 0.5);
    }

    #[test]
    fn test_factual_consistency_stage() {
        let stage = FactualConsistencyStage::new();
        let draft = Draft::new("Paris is the capital of France.", "test");
        let context = create_context();

        let result = stage.verify(&draft, &context);

        assert_eq!(result.stage_name, "factual_consistency");
    }

    #[test]
    fn test_factual_consistency_contradiction() {
        let stage = FactualConsistencyStage::new();
        let draft = Draft::new("Paris is not the capital of France.", "test");
        let context = vec![SearchResult::new(
            Document::new("Paris is the capital of France."),
            0.9,
            0,
        )];

        let result = stage.verify(&draft, &context);

        // Should detect potential contradiction
        assert!(result.notes.is_some());
    }

    #[test]
    fn test_factual_consistency_no_context() {
        let stage = FactualConsistencyStage::new();
        let draft = Draft::new("Some statement", "test");

        let result = stage.verify(&draft, &[]);

        assert_eq!(result.confidence, 0.5);
    }

    #[test]
    fn test_aggregation_average() {
        let pipeline =
            VerificationPipeline::new().with_aggregation_method(AggregationMethod::Average);

        let results = vec![
            StageResult::new("s1", 0.6),
            StageResult::new("s2", 0.8),
            StageResult::new("s3", 0.4),
        ];

        let confidence = pipeline.aggregate_confidence(&results);
        assert!((confidence - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_aggregation_minimum() {
        let pipeline =
            VerificationPipeline::new().with_aggregation_method(AggregationMethod::Minimum);

        let results = vec![
            StageResult::new("s1", 0.6),
            StageResult::new("s2", 0.8),
            StageResult::new("s3", 0.4),
        ];

        let confidence = pipeline.aggregate_confidence(&results);
        assert!((confidence - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_aggregation_maximum() {
        let pipeline =
            VerificationPipeline::new().with_aggregation_method(AggregationMethod::Maximum);

        let results = vec![
            StageResult::new("s1", 0.6),
            StageResult::new("s2", 0.8),
            StageResult::new("s3", 0.4),
        ];

        let confidence = pipeline.aggregate_confidence(&results);
        assert!((confidence - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_aggregation_last_stage() {
        let pipeline =
            VerificationPipeline::new().with_aggregation_method(AggregationMethod::LastStage);

        let results = vec![
            StageResult::new("s1", 0.6),
            StageResult::new("s2", 0.8),
            StageResult::new("s3", 0.4),
        ];

        let confidence = pipeline.aggregate_confidence(&results);
        assert!((confidence - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_aggregation_empty() {
        let pipeline = VerificationPipeline::new();
        let confidence = pipeline.aggregate_confidence(&[]);
        assert_eq!(confidence, 0.5);
    }

    #[test]
    fn test_pipeline_builder() {
        let pipeline = PipelineBuilder::new()
            .with_standard_stages()
            .early_accept(0.9)
            .early_reject(0.1)
            .build();

        assert_eq!(pipeline.num_stages(), 3);
        assert_eq!(pipeline.early_accept_threshold(), 0.9);
        assert_eq!(pipeline.early_reject_threshold(), 0.1);
    }

    #[test]
    fn test_pipeline_builder_individual_stages() {
        let pipeline = PipelineBuilder::new()
            .with_keyword_match()
            .with_semantic_similarity()
            .build();

        assert_eq!(pipeline.num_stages(), 2);
    }

    #[tokio::test]
    async fn test_pipeline_early_accept() {
        let pipeline = VerificationPipeline::new()
            .with_early_accept_threshold(0.5)
            .add_stage(Box::new(KeywordMatchStage::new()));

        let draft = Draft::new(
            "Paris France capital Eiffel Tower Western Europe population million country",
            "test",
        );
        let context = create_context();

        let result = pipeline.verify(&draft, &context).await.unwrap();

        // High keyword overlap should lead to acceptance
        assert!(result.confidence >= 0.5);
    }

    #[test]
    fn test_ngram_extraction() {
        let ngrams = SemanticSimilarityStage::extract_ngrams("hello", 3);
        assert!(ngrams.contains(&"hel".to_string()));
        assert!(ngrams.contains(&"ell".to_string()));
        assert!(ngrams.contains(&"llo".to_string()));
    }

    #[test]
    fn test_ngram_similarity() {
        let sim = SemanticSimilarityStage::ngram_similarity("hello world", "hello world", 3);
        assert_eq!(sim, 1.0);

        let sim2 = SemanticSimilarityStage::ngram_similarity("abc", "xyz", 3);
        assert_eq!(sim2, 0.0);
    }

    #[test]
    fn test_contains_negation() {
        let stage = FactualConsistencyStage::new();
        assert!(stage.contains_negation("This is not correct"));
        assert!(stage.contains_negation("There was never any doubt"));
        assert!(!stage.contains_negation("This is correct"));
    }

    #[test]
    fn test_extract_assertions() {
        let assertions = FactualConsistencyStage::extract_assertions("Paris is the capital.");
        assert!(!assertions.is_empty());
    }

    #[test]
    fn test_pipeline_default() {
        let pipeline = VerificationPipeline::default();
        assert_eq!(pipeline.num_stages(), 0);
    }

    #[test]
    fn test_pipeline_builder_default() {
        let builder = PipelineBuilder::default();
        let pipeline = builder.build();
        assert_eq!(pipeline.num_stages(), 0);
    }

    #[test]
    fn test_aggregation_method_default() {
        let method = AggregationMethod::default();
        assert!(matches!(method, AggregationMethod::Average));
    }
}
