//! Prompt injection detector: pattern scanning, risk scoring, and defense strategies.

use crate::prompt_injection_defense::types::{
    DefenseReport, DefenseStrategy, DocumentScanResult, InjectionCategory, InjectionPattern,
    MatchedPattern, PromptInjectionConfig, PromptInjectionError,
};

// ── PromptInjectionDetector ───────────────────────────────────────────────────

/// Scans retrieved documents for prompt injection attacks and applies the
/// configured [`DefenseStrategy`].
///
/// # Algorithm
///
/// 1. **Pattern matching** — each document is lowercased and scanned for every
///    registered injection pattern using case-insensitive substring search.
///    All non-overlapping occurrences are recorded together with their byte
///    offset, category, and severity.
///
/// 2. **Risk scoring** — the document risk score is
///    `min(1.0, Σ severity_i)` over all matched occurrences.  Multiple hits
///    or a single high-severity hit can push the score to the cap.
///
/// 3. **Defense action** — depends on [`PromptInjectionConfig::strategy`]:
///    - `Quarantine` — documents whose score ≥ threshold are removed from the
///      clean context entirely.
///    - `Sanitize` — matched spans are replaced with `[REDACTED]` in the clean
///      context.
///    - `Flag` — all documents pass through; suspicious ones are marked.
///    - `ScoreOnly` — scores are computed but no action is taken.
///
/// Everything is deterministic and pure Rust (`std` + `thiserror`): no ML,
/// no embeddings, no randomness.
#[derive(Debug, Clone)]
pub struct PromptInjectionDetector {
    /// Configuration for this detector.
    pub config: PromptInjectionConfig,
    patterns: Vec<InjectionPattern>,
}

impl PromptInjectionDetector {
    /// Create a new detector with default built-in patterns.
    #[must_use]
    pub fn new(config: PromptInjectionConfig) -> Self {
        Self {
            config,
            patterns: Self::default_patterns(),
        }
    }

    /// Create a detector with a fully custom pattern set (replaces all defaults).
    #[must_use]
    pub fn with_patterns(config: PromptInjectionConfig, patterns: Vec<InjectionPattern>) -> Self {
        Self { config, patterns }
    }

    /// Append additional patterns to the default set.
    #[must_use]
    pub fn with_extra_patterns(mut self, extra: Vec<InjectionPattern>) -> Self {
        self.patterns.extend(extra);
        self
    }

    /// Return a reference to the active pattern list.
    #[must_use]
    pub fn patterns(&self) -> &[InjectionPattern] {
        &self.patterns
    }

    // ── default patterns ──────────────────────────────────────────────────────

    #[allow(clippy::too_many_lines)]
    fn default_patterns() -> Vec<InjectionPattern> {
        use InjectionCategory::{
            BoundaryInjection, DirectCommand, Exfiltration, Manipulation, RoleOverride,
        };

        vec![
            // ── BoundaryInjection (highest severity) ──────────────────────────
            InjectionPattern {
                text: "<|system|>".to_owned(),
                category: BoundaryInjection,
                severity: 0.90,
            },
            InjectionPattern {
                text: "<|im_start|>system".to_owned(),
                category: BoundaryInjection,
                severity: 0.90,
            },
            InjectionPattern {
                text: "[inst]".to_owned(),
                category: BoundaryInjection,
                severity: 0.90,
            },
            InjectionPattern {
                text: "<<sys>>".to_owned(),
                category: BoundaryInjection,
                severity: 0.90,
            },
            InjectionPattern {
                text: "[system]".to_owned(),
                category: BoundaryInjection,
                severity: 0.85,
            },
            InjectionPattern {
                text: "###system".to_owned(),
                category: BoundaryInjection,
                severity: 0.85,
            },
            InjectionPattern {
                text: "</s>".to_owned(),
                category: BoundaryInjection,
                severity: 0.80,
            },
            InjectionPattern {
                text: "<|endoftext|>".to_owned(),
                category: BoundaryInjection,
                severity: 0.85,
            },
            // ── RoleOverride ──────────────────────────────────────────────────
            InjectionPattern {
                text: "ignore previous instructions".to_owned(),
                category: RoleOverride,
                severity: 0.95,
            },
            InjectionPattern {
                text: "ignore all previous".to_owned(),
                category: RoleOverride,
                severity: 0.90,
            },
            InjectionPattern {
                text: "disregard previous".to_owned(),
                category: RoleOverride,
                severity: 0.85,
            },
            InjectionPattern {
                text: "forget your instructions".to_owned(),
                category: RoleOverride,
                severity: 0.90,
            },
            InjectionPattern {
                text: "override instructions".to_owned(),
                category: RoleOverride,
                severity: 0.90,
            },
            InjectionPattern {
                text: "you are now".to_owned(),
                category: RoleOverride,
                severity: 0.75,
            },
            InjectionPattern {
                text: "your new role".to_owned(),
                category: RoleOverride,
                severity: 0.80,
            },
            InjectionPattern {
                text: "pretend you are".to_owned(),
                category: RoleOverride,
                severity: 0.75,
            },
            InjectionPattern {
                text: "you must now".to_owned(),
                category: RoleOverride,
                severity: 0.70,
            },
            InjectionPattern {
                text: "new instructions".to_owned(),
                category: RoleOverride,
                severity: 0.70,
            },
            InjectionPattern {
                text: "updated instructions".to_owned(),
                category: RoleOverride,
                severity: 0.65,
            },
            InjectionPattern {
                text: "system prompt".to_owned(),
                category: RoleOverride,
                severity: 0.60,
            },
            // ── Exfiltration ──────────────────────────────────────────────────
            InjectionPattern {
                text: "reveal your system prompt".to_owned(),
                category: Exfiltration,
                severity: 0.95,
            },
            InjectionPattern {
                text: "print your instructions".to_owned(),
                category: Exfiltration,
                severity: 0.90,
            },
            InjectionPattern {
                text: "repeat the above".to_owned(),
                category: Exfiltration,
                severity: 0.85,
            },
            InjectionPattern {
                text: "repeat everything above".to_owned(),
                category: Exfiltration,
                severity: 0.90,
            },
            InjectionPattern {
                text: "show your prompt".to_owned(),
                category: Exfiltration,
                severity: 0.85,
            },
            InjectionPattern {
                text: "what are your instructions".to_owned(),
                category: Exfiltration,
                severity: 0.80,
            },
            InjectionPattern {
                text: "output your system".to_owned(),
                category: Exfiltration,
                severity: 0.85,
            },
            // ── DirectCommand ─────────────────────────────────────────────────
            InjectionPattern {
                text: "your real task is".to_owned(),
                category: DirectCommand,
                severity: 0.80,
            },
            InjectionPattern {
                text: "the real instructions are".to_owned(),
                category: DirectCommand,
                severity: 0.85,
            },
            InjectionPattern {
                text: "your actual task".to_owned(),
                category: DirectCommand,
                severity: 0.75,
            },
            InjectionPattern {
                text: "instead, you should".to_owned(),
                category: DirectCommand,
                severity: 0.60,
            },
            InjectionPattern {
                text: "however, you should".to_owned(),
                category: DirectCommand,
                severity: 0.55,
            },
            InjectionPattern {
                text: "do not follow".to_owned(),
                category: DirectCommand,
                severity: 0.70,
            },
            InjectionPattern {
                text: "stop following".to_owned(),
                category: DirectCommand,
                severity: 0.75,
            },
            // ── Manipulation ──────────────────────────────────────────────────
            InjectionPattern {
                text: "jailbreak".to_owned(),
                category: Manipulation,
                severity: 0.85,
            },
            InjectionPattern {
                text: "dan mode".to_owned(),
                category: Manipulation,
                severity: 0.90,
            },
            InjectionPattern {
                text: "developer mode".to_owned(),
                category: Manipulation,
                severity: 0.80,
            },
            InjectionPattern {
                text: "from anthropic".to_owned(),
                category: Manipulation,
                severity: 0.70,
            },
            InjectionPattern {
                text: "from openai".to_owned(),
                category: Manipulation,
                severity: 0.70,
            },
            InjectionPattern {
                text: "from the developers".to_owned(),
                category: Manipulation,
                severity: 0.60,
            },
            InjectionPattern {
                text: "this is a test".to_owned(),
                category: Manipulation,
                severity: 0.40,
            },
            InjectionPattern {
                text: "trust me".to_owned(),
                category: Manipulation,
                severity: 0.35,
            },
        ]
    }

    // ── internal scanning ─────────────────────────────────────────────────────

    /// Scan a single document and return all matches plus the risk score.
    ///
    /// Matching is case-insensitive substring search.  All built-in patterns
    /// are pure ASCII, so byte offsets in the lowercased copy equal those in
    /// the original document.
    fn scan_document(&self, document: &str) -> (Vec<MatchedPattern>, f32) {
        let doc_lower = document.to_lowercase();
        let mut matches: Vec<MatchedPattern> = Vec::new();

        for pattern in &self.patterns {
            let needle = pattern.text.to_lowercase();
            let needle_len = needle.len();
            let mut search_start = 0;

            while search_start + needle_len <= doc_lower.len() {
                match doc_lower[search_start..].find(&needle) {
                    Some(rel) => {
                        let byte_offset = search_start + rel;
                        matches.push(MatchedPattern {
                            text: pattern.text.clone(),
                            category: pattern.category.clone(),
                            severity: pattern.severity,
                            byte_offset,
                        });
                        search_start = byte_offset + needle_len;
                    }
                    None => break,
                }
            }
        }

        matches.sort_by_key(|m| m.byte_offset);

        let total_severity: f32 = matches.iter().map(|m| m.severity).sum();
        let risk_score = total_severity.min(1.0);

        (matches, risk_score)
    }

    /// Replace all matched byte spans with `[REDACTED]` in the original document.
    ///
    /// Overlapping spans are merged before replacement.  Since all patterns are
    /// ASCII (1 byte per char), the byte offsets are valid UTF-8 boundaries in
    /// the original document.
    fn sanitize(document: &str, matched: &[MatchedPattern]) -> String {
        if matched.is_empty() {
            return document.to_owned();
        }

        // Build and sort spans; span end = offset + pattern length in bytes.
        let mut spans: Vec<(usize, usize)> = matched
            .iter()
            .map(|m| {
                let end = (m.byte_offset + m.text.len()).min(document.len());
                (m.byte_offset, end)
            })
            .collect();
        spans.sort_by_key(|&(s, _)| s);

        // Merge overlapping / adjacent spans.
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(spans.len());
        for (s, e) in spans {
            match merged.last_mut() {
                Some(last) if s <= last.1 => last.1 = last.1.max(e),
                _ => merged.push((s, e)),
            }
        }

        let mut result = String::with_capacity(document.len());
        let mut cursor = 0;
        for (start, end) in merged {
            if start > cursor {
                result.push_str(&document[cursor..start]);
            }
            result.push_str("[REDACTED]");
            cursor = end;
        }
        if cursor < document.len() {
            result.push_str(&document[cursor..]);
        }
        result
    }

    /// Scan one document and produce its [`DocumentScanResult`].
    fn scan_single(&self, document: &str) -> DocumentScanResult {
        let (matched_patterns, risk_score) = self.scan_document(document);
        let is_suspicious = risk_score > 0.0;

        let is_quarantined = matches!(self.config.strategy, DefenseStrategy::Quarantine)
            && risk_score >= self.config.quarantine_threshold;

        let sanitized = if matches!(self.config.strategy, DefenseStrategy::Sanitize)
            && !matched_patterns.is_empty()
        {
            Some(Self::sanitize(document, &matched_patterns))
        } else {
            None
        };

        DocumentScanResult {
            document: document.to_owned(),
            risk_score,
            matched_patterns,
            is_suspicious,
            is_quarantined,
            sanitized,
        }
    }

    // ── public API ────────────────────────────────────────────────────────────

    /// Scan all documents in `context` and return a [`DefenseReport`].
    ///
    /// # Errors
    ///
    /// * [`PromptInjectionError::EmptyContext`] — when `context` is empty.
    #[allow(clippy::cast_precision_loss)]
    pub fn scan(&self, context: &[String]) -> Result<DefenseReport, PromptInjectionError> {
        if context.is_empty() {
            return Err(PromptInjectionError::EmptyContext);
        }

        let results: Vec<DocumentScanResult> =
            context.iter().map(|doc| self.scan_single(doc)).collect();

        let quarantined_count = results.iter().filter(|r| r.is_quarantined).count();
        let suspicious_count = results.iter().filter(|r| r.is_suspicious).count();
        let overall_risk = results.iter().map(|r| r.risk_score).fold(0.0_f32, f32::max);

        let clean_context: Vec<String> = results
            .iter()
            .filter_map(|r| {
                if r.is_quarantined {
                    return None;
                }
                Some(r.sanitized.clone().unwrap_or_else(|| r.document.clone()))
            })
            .collect();

        Ok(DefenseReport {
            total_documents: results.len(),
            quarantined_count,
            suspicious_count,
            overall_risk,
            clean_context,
            results,
        })
    }
}
