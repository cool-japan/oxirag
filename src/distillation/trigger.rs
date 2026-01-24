//! Distillation trigger conditions for automatic model training.
//!
//! This module provides trigger conditions that determine when
//! a query pattern is ready for distillation into a specialized model.

use super::types::{DistillationCandidate, QueryPattern, current_timestamp};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Trigger condition for starting distillation.
///
/// Multiple conditions can be combined to create complex triggering logic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TriggerCondition {
    /// Trigger when frequency reaches a threshold.
    FrequencyThreshold(u32),
    /// Trigger when Q&A pair count reaches a minimum.
    QAPairCount(usize),
    /// Trigger when average confidence falls below a threshold.
    ConfidenceBelow(f32),
    /// Trigger when average confidence is above a threshold.
    ConfidenceAbove(f32),
    /// Trigger after a certain time since first seen (in seconds).
    TimeSinceFirstSeen(u64),
    /// Trigger when a combination of conditions are met (AND logic).
    Combined(Vec<TriggerCondition>),
    /// Trigger when any of the conditions are met (OR logic).
    Any(Vec<TriggerCondition>),
}

impl TriggerCondition {
    /// Create a frequency threshold condition.
    #[must_use]
    pub fn frequency(threshold: u32) -> Self {
        Self::FrequencyThreshold(threshold)
    }

    /// Create a Q&A pair count condition.
    #[must_use]
    pub fn min_qa_pairs(count: usize) -> Self {
        Self::QAPairCount(count)
    }

    /// Create a confidence below threshold condition.
    #[must_use]
    pub fn confidence_below(threshold: f32) -> Self {
        Self::ConfidenceBelow(threshold.clamp(0.0, 1.0))
    }

    /// Create a confidence above threshold condition.
    #[must_use]
    pub fn confidence_above(threshold: f32) -> Self {
        Self::ConfidenceAbove(threshold.clamp(0.0, 1.0))
    }

    /// Create a time-since-first-seen condition.
    #[must_use]
    pub fn time_since_first_seen(duration: Duration) -> Self {
        Self::TimeSinceFirstSeen(duration.as_secs())
    }

    /// Combine multiple conditions with AND logic.
    #[must_use]
    pub fn all(conditions: Vec<TriggerCondition>) -> Self {
        Self::Combined(conditions)
    }

    /// Combine multiple conditions with OR logic.
    #[must_use]
    pub fn any_of(conditions: Vec<TriggerCondition>) -> Self {
        Self::Any(conditions)
    }

    /// Check if this condition is satisfied by a candidate.
    #[must_use]
    pub fn is_satisfied(&self, candidate: &DistillationCandidate) -> bool {
        match self {
            Self::FrequencyThreshold(threshold) => candidate.frequency >= *threshold,
            Self::QAPairCount(min_count) => candidate.qa_pairs.len() >= *min_count,
            Self::ConfidenceBelow(threshold) => {
                !candidate.qa_pairs.is_empty() && candidate.avg_confidence < *threshold
            }
            Self::ConfidenceAbove(threshold) => {
                !candidate.qa_pairs.is_empty() && candidate.avg_confidence >= *threshold
            }
            Self::TimeSinceFirstSeen(secs) => {
                let now = current_timestamp();
                now.saturating_sub(candidate.first_seen) >= *secs
            }
            Self::Combined(conditions) => conditions.iter().all(|c| c.is_satisfied(candidate)),
            Self::Any(conditions) => conditions.iter().any(|c| c.is_satisfied(candidate)),
        }
    }

    /// Get a human-readable description of this condition.
    #[must_use]
    pub fn description(&self) -> String {
        match self {
            Self::FrequencyThreshold(t) => format!("frequency >= {t}"),
            Self::QAPairCount(c) => format!("Q&A pairs >= {c}"),
            Self::ConfidenceBelow(t) => format!("confidence < {t:.2}"),
            Self::ConfidenceAbove(t) => format!("confidence >= {t:.2}"),
            Self::TimeSinceFirstSeen(s) => format!("time since first seen >= {s}s"),
            Self::Combined(conditions) => {
                let descs: Vec<_> = conditions.iter().map(Self::description).collect();
                format!("({})", descs.join(" AND "))
            }
            Self::Any(conditions) => {
                let descs: Vec<_> = conditions.iter().map(Self::description).collect();
                format!("({})", descs.join(" OR "))
            }
        }
    }
}

impl Default for TriggerCondition {
    fn default() -> Self {
        // Default: frequency >= 5 AND Q&A pairs >= 3 AND confidence >= 0.7
        Self::Combined(vec![
            Self::FrequencyThreshold(5),
            Self::QAPairCount(3),
            Self::ConfidenceAbove(0.7),
        ])
    }
}

/// Distillation trigger manager.
///
/// Manages trigger conditions and cooldown periods to determine
/// when patterns should be distilled.
#[derive(Debug)]
pub struct DistillationTrigger {
    /// The conditions that must be met for triggering.
    conditions: Vec<TriggerCondition>,
    /// Cooldown period between triggers for the same pattern.
    cooldown: Duration,
    /// Map from pattern hash to last trigger timestamp.
    last_triggered: HashMap<u64, u64>,
}

impl DistillationTrigger {
    /// Create a new trigger with the given conditions.
    #[must_use]
    pub fn new(conditions: Vec<TriggerCondition>) -> Self {
        Self {
            conditions,
            cooldown: Duration::from_secs(3600), // 1 hour default
            last_triggered: HashMap::new(),
        }
    }

    /// Create a trigger with a single condition.
    #[must_use]
    pub fn with_condition(condition: TriggerCondition) -> Self {
        Self::new(vec![condition])
    }

    /// Create a trigger with default conditions.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(vec![TriggerCondition::default()])
    }

    /// Set the cooldown period.
    #[must_use]
    pub fn with_cooldown(mut self, cooldown: Duration) -> Self {
        self.cooldown = cooldown;
        self
    }

    /// Add an additional condition.
    #[must_use]
    pub fn with_additional_condition(mut self, condition: TriggerCondition) -> Self {
        self.conditions.push(condition);
        self
    }

    /// Get the current conditions.
    #[must_use]
    pub fn conditions(&self) -> &[TriggerCondition] {
        &self.conditions
    }

    /// Get the cooldown duration.
    #[must_use]
    pub fn cooldown(&self) -> Duration {
        self.cooldown
    }

    /// Check if a candidate should trigger distillation.
    ///
    /// Returns true if all conditions are met and the cooldown has passed.
    #[must_use]
    pub fn should_trigger(&self, candidate: &DistillationCandidate) -> bool {
        // Check cooldown
        if self.is_in_cooldown(&candidate.pattern) {
            return false;
        }

        // Check all conditions
        self.conditions.iter().all(|c| c.is_satisfied(candidate))
    }

    /// Check if a pattern is currently in cooldown.
    #[must_use]
    pub fn is_in_cooldown(&self, pattern: &QueryPattern) -> bool {
        if let Some(last_time) = self.last_triggered.get(&pattern.pattern_hash) {
            let now = current_timestamp();
            now.saturating_sub(*last_time) < self.cooldown.as_secs()
        } else {
            false
        }
    }

    /// Get the remaining cooldown time for a pattern.
    #[must_use]
    pub fn remaining_cooldown(&self, pattern: &QueryPattern) -> Option<Duration> {
        self.last_triggered
            .get(&pattern.pattern_hash)
            .and_then(|last_time| {
                let now = current_timestamp();
                let elapsed = now.saturating_sub(*last_time);
                let cooldown_secs = self.cooldown.as_secs();
                if elapsed < cooldown_secs {
                    Some(Duration::from_secs(cooldown_secs - elapsed))
                } else {
                    None
                }
            })
    }

    /// Mark a pattern as triggered.
    pub fn mark_triggered(&mut self, pattern: &QueryPattern) {
        self.last_triggered
            .insert(pattern.pattern_hash, current_timestamp());
    }

    /// Clear the triggered status for a pattern.
    pub fn clear_triggered(&mut self, pattern: &QueryPattern) {
        self.last_triggered.remove(&pattern.pattern_hash);
    }

    /// Evaluate all candidates and return those that should trigger.
    #[must_use]
    pub fn evaluate_all<'a>(
        &self,
        candidates: &'a [DistillationCandidate],
    ) -> Vec<&'a DistillationCandidate> {
        candidates
            .iter()
            .filter(|c| self.should_trigger(c))
            .collect()
    }

    /// Evaluate all candidates and return results with evaluation details.
    #[must_use]
    pub fn evaluate_with_details<'a>(
        &self,
        candidates: &'a [DistillationCandidate],
    ) -> Vec<TriggerEvaluation<'a>> {
        candidates
            .iter()
            .map(|c| self.evaluate_candidate(c))
            .collect()
    }

    /// Evaluate a single candidate with detailed results.
    #[must_use]
    pub fn evaluate_candidate<'a>(
        &self,
        candidate: &'a DistillationCandidate,
    ) -> TriggerEvaluation<'a> {
        let in_cooldown = self.is_in_cooldown(&candidate.pattern);
        let condition_results: Vec<_> = self
            .conditions
            .iter()
            .map(|c| (c.description(), c.is_satisfied(candidate)))
            .collect();

        let all_conditions_met = condition_results.iter().all(|(_, met)| *met);
        let should_trigger = all_conditions_met && !in_cooldown;

        TriggerEvaluation {
            candidate,
            should_trigger,
            in_cooldown,
            remaining_cooldown: self.remaining_cooldown(&candidate.pattern),
            condition_results,
        }
    }

    /// Get statistics about trigger state.
    #[must_use]
    pub fn statistics(&self) -> TriggerStatistics {
        let now = current_timestamp();
        let cooldown_secs = self.cooldown.as_secs();

        let patterns_in_cooldown = self
            .last_triggered
            .values()
            .filter(|&t| now.saturating_sub(*t) < cooldown_secs)
            .count();

        TriggerStatistics {
            total_conditions: self.conditions.len(),
            patterns_tracked: self.last_triggered.len(),
            patterns_in_cooldown,
            cooldown_secs: self.cooldown.as_secs(),
        }
    }

    /// Clear all trigger history.
    pub fn clear_history(&mut self) {
        self.last_triggered.clear();
    }

    /// Prune old entries from the trigger history.
    pub fn prune_history(&mut self, max_age_secs: u64) {
        let now = current_timestamp();
        self.last_triggered
            .retain(|_, t| now.saturating_sub(*t) <= max_age_secs);
    }
}

impl Default for DistillationTrigger {
    fn default() -> Self {
        Self::with_defaults()
    }
}

impl Clone for DistillationTrigger {
    fn clone(&self) -> Self {
        Self {
            conditions: self.conditions.clone(),
            cooldown: self.cooldown,
            last_triggered: self.last_triggered.clone(),
        }
    }
}

/// Detailed evaluation result for a candidate.
#[derive(Debug)]
pub struct TriggerEvaluation<'a> {
    /// The evaluated candidate.
    pub candidate: &'a DistillationCandidate,
    /// Whether the candidate should trigger distillation.
    pub should_trigger: bool,
    /// Whether the candidate is in cooldown.
    pub in_cooldown: bool,
    /// Remaining cooldown time (if any).
    pub remaining_cooldown: Option<Duration>,
    /// Results for each condition (description, satisfied).
    pub condition_results: Vec<(String, bool)>,
}

impl TriggerEvaluation<'_> {
    /// Get the number of conditions that are satisfied.
    #[must_use]
    pub fn satisfied_count(&self) -> usize {
        self.condition_results
            .iter()
            .filter(|(_, met)| *met)
            .count()
    }

    /// Get the total number of conditions.
    #[must_use]
    pub fn total_conditions(&self) -> usize {
        self.condition_results.len()
    }

    /// Get the percentage of conditions satisfied.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn satisfaction_ratio(&self) -> f32 {
        if self.condition_results.is_empty() {
            1.0
        } else {
            self.satisfied_count() as f32 / self.total_conditions() as f32
        }
    }
}

/// Statistics about the trigger state.
#[derive(Debug, Clone, Default)]
pub struct TriggerStatistics {
    /// Total number of conditions configured.
    pub total_conditions: usize,
    /// Number of patterns being tracked.
    pub patterns_tracked: usize,
    /// Number of patterns currently in cooldown.
    pub patterns_in_cooldown: usize,
    /// Cooldown duration in seconds.
    pub cooldown_secs: u64,
}

#[cfg(test)]
#[allow(clippy::similar_names)]
mod tests {
    use super::*;
    use crate::distillation::types::QAPair;

    fn create_test_candidate(
        query: &str,
        frequency: u32,
        pairs_count: usize,
        confidence: f32,
    ) -> DistillationCandidate {
        let pattern = QueryPattern::new(query);
        let mut candidate = DistillationCandidate::new(pattern.clone());

        // Set frequency
        for _ in 1..frequency {
            candidate.record_occurrence();
        }

        // Add Q&A pairs
        for i in 0..pairs_count {
            let pair = QAPair::new(query, &format!("Answer {i}"), confidence, pattern.clone());
            candidate.add_qa_pair(pair, 100);
        }

        candidate
    }

    #[test]
    fn test_frequency_threshold_condition() {
        let condition = TriggerCondition::frequency(5);

        let low = create_test_candidate("test", 3, 1, 0.9);
        let high = create_test_candidate("test", 7, 1, 0.9);

        assert!(!condition.is_satisfied(&low));
        assert!(condition.is_satisfied(&high));
    }

    #[test]
    fn test_qa_pair_count_condition() {
        let condition = TriggerCondition::min_qa_pairs(3);

        let few = create_test_candidate("test", 5, 2, 0.9);
        let many = create_test_candidate("test", 5, 5, 0.9);

        assert!(!condition.is_satisfied(&few));
        assert!(condition.is_satisfied(&many));
    }

    #[test]
    fn test_confidence_below_condition() {
        let condition = TriggerCondition::confidence_below(0.8);

        let high_conf = create_test_candidate("test", 5, 2, 0.9);
        let low_conf = create_test_candidate("test", 5, 2, 0.7);

        assert!(!condition.is_satisfied(&high_conf));
        assert!(condition.is_satisfied(&low_conf));
    }

    #[test]
    fn test_confidence_above_condition() {
        let condition = TriggerCondition::confidence_above(0.8);

        let high_conf = create_test_candidate("test", 5, 2, 0.9);
        let low_conf = create_test_candidate("test", 5, 2, 0.7);

        assert!(condition.is_satisfied(&high_conf));
        assert!(!condition.is_satisfied(&low_conf));
    }

    #[test]
    fn test_combined_conditions() {
        let condition = TriggerCondition::all(vec![
            TriggerCondition::frequency(5),
            TriggerCondition::min_qa_pairs(2),
        ]);

        let meets_none = create_test_candidate("test", 3, 1, 0.9);
        let meets_one = create_test_candidate("test", 6, 1, 0.9);
        let meets_all = create_test_candidate("test", 6, 3, 0.9);

        assert!(!condition.is_satisfied(&meets_none));
        assert!(!condition.is_satisfied(&meets_one));
        assert!(condition.is_satisfied(&meets_all));
    }

    #[test]
    fn test_any_conditions() {
        let condition = TriggerCondition::any_of(vec![
            TriggerCondition::frequency(10),
            TriggerCondition::min_qa_pairs(5),
        ]);

        let meets_none = create_test_candidate("test", 3, 2, 0.9);
        let meets_freq = create_test_candidate("test", 12, 2, 0.9);
        let meets_pairs = create_test_candidate("test", 3, 6, 0.9);

        assert!(!condition.is_satisfied(&meets_none));
        assert!(condition.is_satisfied(&meets_freq));
        assert!(condition.is_satisfied(&meets_pairs));
    }

    #[test]
    fn test_condition_description() {
        let freq = TriggerCondition::frequency(5);
        assert_eq!(freq.description(), "frequency >= 5");

        let combined = TriggerCondition::all(vec![
            TriggerCondition::frequency(5),
            TriggerCondition::min_qa_pairs(3),
        ]);
        assert!(combined.description().contains("AND"));
    }

    #[test]
    fn test_distillation_trigger_creation() {
        let trigger = DistillationTrigger::with_defaults();
        assert!(!trigger.conditions.is_empty());
    }

    #[test]
    fn test_distillation_trigger_with_cooldown() {
        let trigger = DistillationTrigger::with_defaults().with_cooldown(Duration::from_secs(1800));

        assert_eq!(trigger.cooldown().as_secs(), 1800);
    }

    #[test]
    fn test_distillation_trigger_should_trigger() {
        let trigger = DistillationTrigger::new(vec![
            TriggerCondition::frequency(3),
            TriggerCondition::min_qa_pairs(2),
        ]);

        let ready = create_test_candidate("test", 5, 3, 0.9);
        let not_ready = create_test_candidate("test", 2, 1, 0.9);

        assert!(trigger.should_trigger(&ready));
        assert!(!trigger.should_trigger(&not_ready));
    }

    #[test]
    fn test_distillation_trigger_cooldown() {
        let mut trigger = DistillationTrigger::new(vec![TriggerCondition::frequency(3)])
            .with_cooldown(Duration::from_secs(3600));

        let candidate = create_test_candidate("test", 5, 1, 0.9);

        // Should trigger initially
        assert!(trigger.should_trigger(&candidate));

        // Mark as triggered
        trigger.mark_triggered(&candidate.pattern);

        // Should not trigger due to cooldown
        assert!(!trigger.should_trigger(&candidate));
        assert!(trigger.is_in_cooldown(&candidate.pattern));
    }

    #[test]
    fn test_distillation_trigger_clear_triggered() {
        let mut trigger = DistillationTrigger::new(vec![TriggerCondition::frequency(3)])
            .with_cooldown(Duration::from_secs(3600));

        let candidate = create_test_candidate("test", 5, 1, 0.9);

        trigger.mark_triggered(&candidate.pattern);
        assert!(trigger.is_in_cooldown(&candidate.pattern));

        trigger.clear_triggered(&candidate.pattern);
        assert!(!trigger.is_in_cooldown(&candidate.pattern));
    }

    #[test]
    fn test_distillation_trigger_evaluate_all() {
        let trigger = DistillationTrigger::new(vec![TriggerCondition::frequency(5)]);

        let candidates = vec![
            create_test_candidate("ready1", 6, 1, 0.9),
            create_test_candidate("not_ready", 3, 1, 0.9),
            create_test_candidate("ready2", 8, 1, 0.9),
        ];

        let triggered = trigger.evaluate_all(&candidates);
        assert_eq!(triggered.len(), 2);
    }

    #[test]
    fn test_distillation_trigger_evaluate_with_details() {
        let trigger = DistillationTrigger::new(vec![
            TriggerCondition::frequency(5),
            TriggerCondition::min_qa_pairs(2),
        ]);

        let candidate = create_test_candidate("test", 3, 1, 0.9);
        let eval = trigger.evaluate_candidate(&candidate);

        assert!(!eval.should_trigger);
        assert_eq!(eval.total_conditions(), 2);
        assert_eq!(eval.satisfied_count(), 0);
    }

    #[test]
    fn test_trigger_evaluation_satisfaction_ratio() {
        let trigger = DistillationTrigger::new(vec![
            TriggerCondition::frequency(3),
            TriggerCondition::min_qa_pairs(5),
        ]);

        let candidate = create_test_candidate("test", 5, 3, 0.9);
        let eval = trigger.evaluate_candidate(&candidate);

        // Frequency met, pairs not met = 50%
        assert!((eval.satisfaction_ratio() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_distillation_trigger_statistics() {
        let mut trigger = DistillationTrigger::new(vec![
            TriggerCondition::frequency(3),
            TriggerCondition::min_qa_pairs(2),
        ])
        .with_cooldown(Duration::from_secs(3600));

        let candidate1 = create_test_candidate("test1", 5, 1, 0.9);
        let candidate2 = create_test_candidate("test2", 5, 1, 0.9);

        trigger.mark_triggered(&candidate1.pattern);
        trigger.mark_triggered(&candidate2.pattern);

        let stats = trigger.statistics();
        assert_eq!(stats.total_conditions, 2);
        assert_eq!(stats.patterns_tracked, 2);
        assert_eq!(stats.patterns_in_cooldown, 2);
    }

    #[test]
    fn test_distillation_trigger_clear_history() {
        let mut trigger = DistillationTrigger::with_defaults();

        let candidate = create_test_candidate("test", 5, 1, 0.9);
        trigger.mark_triggered(&candidate.pattern);

        assert!(!trigger.last_triggered.is_empty());

        trigger.clear_history();
        assert!(trigger.last_triggered.is_empty());
    }

    #[test]
    fn test_distillation_trigger_clone() {
        let mut trigger = DistillationTrigger::new(vec![TriggerCondition::frequency(5)]);
        let candidate = create_test_candidate("test", 5, 1, 0.9);
        trigger.mark_triggered(&candidate.pattern);

        let cloned = trigger.clone();
        assert_eq!(trigger.conditions.len(), cloned.conditions.len());
        assert!(cloned.is_in_cooldown(&candidate.pattern));
    }

    #[test]
    fn test_default_trigger_condition() {
        let default = TriggerCondition::default();

        // Default is Combined with frequency, pairs, and confidence
        if let TriggerCondition::Combined(conditions) = default {
            assert_eq!(conditions.len(), 3);
        } else {
            panic!("Default should be Combined");
        }
    }

    #[test]
    fn test_with_additional_condition() {
        let trigger = DistillationTrigger::with_defaults()
            .with_additional_condition(TriggerCondition::frequency(10));

        assert_eq!(trigger.conditions.len(), 2);
    }

    #[test]
    fn test_remaining_cooldown() {
        let mut trigger =
            DistillationTrigger::with_defaults().with_cooldown(Duration::from_secs(3600));

        let candidate = create_test_candidate("test", 5, 1, 0.9);

        // No cooldown initially
        assert!(trigger.remaining_cooldown(&candidate.pattern).is_none());

        // After trigger
        trigger.mark_triggered(&candidate.pattern);
        let remaining = trigger.remaining_cooldown(&candidate.pattern);
        assert!(remaining.is_some());
        assert!(remaining.unwrap().as_secs() > 0);
    }

    #[test]
    fn test_time_since_first_seen_condition() {
        let condition = TriggerCondition::time_since_first_seen(Duration::from_secs(0));

        let candidate = create_test_candidate("test", 5, 1, 0.9);

        // Should be satisfied immediately (time_since = 0)
        assert!(condition.is_satisfied(&candidate));
    }

    #[test]
    fn test_condition_with_no_qa_pairs() {
        let confidence_below = TriggerCondition::confidence_below(0.5);
        let confidence_above = TriggerCondition::confidence_above(0.5);

        let candidate = create_test_candidate("test", 5, 0, 0.0);

        // Neither should be satisfied with no Q&A pairs
        assert!(!confidence_below.is_satisfied(&candidate));
        assert!(!confidence_above.is_satisfied(&candidate));
    }
}
