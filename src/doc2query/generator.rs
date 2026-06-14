//! Heuristic hypothetical-query generation.
use crate::doc2query::types::{Doc2QueryConfig, QueryGenerator};
use crate::types::Document;
use std::collections::{HashMap, HashSet};

// ── stopwords ─────────────────────────────────────────────────────────────────

/// Small English stopword set excluded from salient-term ranking.
const STOPWORDS: &[&str] = &[
    "the", "and", "for", "are", "was", "were", "with", "that", "this", "from", "have", "has",
    "had", "not", "but", "you", "all", "can", "her", "his", "its", "our", "out", "use", "any",
    "how", "who", "why", "they", "them", "their", "what", "when", "which", "into", "than", "then",
    "some", "such", "only", "also", "been", "more", "most", "other", "over", "these", "those",
    "will", "would", "could", "should", "about", "between", "because", "while", "does", "did",
    "doing", "where", "there", "here", "very", "just", "each",
];

// ── tokenizer helpers ─────────────────────────────────────────────────────────

/// Tokenize text into lowercase alphanumeric tokens of length >= 2.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(str::to_lowercase)
        .collect()
}

/// Split text into sentences on `.`, `!` and `?`.
fn sentences(text: &str) -> Vec<String> {
    text.split(['.', '!', '?'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

/// Whether a raw word looks like a proper-noun / entity token.
///
/// True when the word starts with an uppercase letter and the remainder is
/// lowercase alphabetic (avoids ALL-CAPS acronyms and mixed-case noise).
fn is_entity_word(word: &str) -> bool {
    let mut chars = word.chars();
    match chars.next() {
        Some(first) if first.is_uppercase() => {
            let rest: String = chars.collect();
            !rest.is_empty() && rest.chars().all(|c| c.is_lowercase() && c.is_alphabetic())
        }
        _ => false,
    }
}

// ── HeuristicQueryGenerator ───────────────────────────────────────────────────

/// Generates hypothetical questions a passage answers using lexical heuristics.
///
/// The generator ranks salient terms by term frequency (excluding a small
/// stopword set), detects capitalized entity phrases, and emits deterministic
/// template questions. No external model or randomness is involved, so output is
/// fully reproducible and every emitted token originates from the document.
#[derive(Debug, Clone, Default)]
pub struct HeuristicQueryGenerator {
    /// Generation configuration.
    pub config: Doc2QueryConfig,
}

impl HeuristicQueryGenerator {
    /// Create a new generator with the given configuration.
    #[must_use]
    pub fn new(config: Doc2QueryConfig) -> Self {
        Self { config }
    }

    /// Combine the document title and content into a single source string.
    fn source_text(doc: &Document) -> String {
        match &doc.title {
            Some(title) if !title.trim().is_empty() => {
                format!("{title}. {}", doc.content)
            }
            _ => doc.content.clone(),
        }
    }

    /// Rank salient content terms by term frequency, descending.
    ///
    /// Ties are broken alphabetically so the ordering is deterministic.
    fn salient_terms(text: &str) -> Vec<String> {
        let mut tf: HashMap<String, usize> = HashMap::new();
        for tok in tokenize(text) {
            if STOPWORDS.contains(&tok.as_str()) || tok.len() < 3 {
                continue;
            }
            *tf.entry(tok).or_insert(0) += 1;
        }
        let mut ranked: Vec<(String, usize)> = tf.into_iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked.into_iter().map(|(term, _)| term).collect()
    }

    /// Detect capitalized entity phrases in document order (deduplicated).
    fn entities(text: &str) -> Vec<String> {
        let mut seen: HashSet<String> = HashSet::new();
        let mut entities: Vec<String> = Vec::new();
        for raw in text.split(|c: char| !c.is_alphanumeric()) {
            if is_entity_word(raw) && !STOPWORDS.contains(&raw.to_lowercase().as_str()) {
                let key = raw.to_lowercase();
                if seen.insert(key) {
                    entities.push(raw.to_string());
                }
            }
        }
        entities
    }

    /// Build the definitional question family over salient terms, in order.
    fn definitional_questions(terms: &[String]) -> Vec<String> {
        let mut questions = Vec::with_capacity(terms.len() * 2);
        for term in terms {
            questions.push(format!("What is {term}?"));
            questions.push(format!("What does {term} mean?"));
        }
        questions
    }

    /// Build the relational question family over entity pairs, in document order.
    fn relational_questions(entities: &[String]) -> Vec<String> {
        let mut questions = Vec::new();
        for i in 0..entities.len() {
            for j in (i + 1)..entities.len() {
                let a = &entities[i];
                let b = &entities[j];
                questions.push(format!("How does {a} relate to {b}?"));
                questions.push(format!("What is the connection between {a} and {b}?"));
            }
        }
        questions
    }

    /// Build the factoid question family, one per sentence subject, in order.
    fn factoid_questions(sents: &[String]) -> Vec<String> {
        sents
            .iter()
            .filter_map(|sentence| Self::salient_terms(sentence).into_iter().next())
            .map(|subject| format!("Which details concern {subject}?"))
            .collect()
    }

    /// Drain questions from `family` into `out` (deduped) until `out` reaches `cap`.
    fn drain_into(
        out: &mut Vec<String>,
        seen: &mut HashSet<String>,
        family: &[String],
        cap: usize,
    ) {
        for question in family {
            if out.len() >= cap {
                return;
            }
            if seen.insert(question.to_lowercase()) {
                out.push(question.clone());
            }
        }
    }
}

impl QueryGenerator for HeuristicQueryGenerator {
    fn generate(&self, doc: &Document, n: usize) -> Vec<String> {
        if n == 0 {
            return Vec::new();
        }
        let text = Self::source_text(doc);
        let terms = Self::salient_terms(&text);
        let entities = Self::entities(&text);
        let sents = sentences(&text);

        let definitional = if self.config.include_definitional {
            Self::definitional_questions(&terms)
        } else {
            Vec::new()
        };
        let relational = if self.config.include_relational && entities.len() >= 2 {
            Self::relational_questions(&entities)
        } else {
            Vec::new()
        };
        let factoid = Self::factoid_questions(&sents);

        let mut out: Vec<String> = Vec::with_capacity(n);
        let mut seen: HashSet<String> = HashSet::new();

        // When both lead families are active, reserve part of the budget for
        // relational questions so definitional terms cannot crowd them out.
        // Ordering stays deterministic: definitional first (up to its cap),
        // then relational, then any remaining definitional, then factoid.
        if !definitional.is_empty() && !relational.is_empty() {
            let reserved = (n / 2).max(1).min(relational.len());
            let definitional_cap = n.saturating_sub(reserved);
            Self::drain_into(&mut out, &mut seen, &definitional, definitional_cap);
            Self::drain_into(&mut out, &mut seen, &relational, n);
            Self::drain_into(&mut out, &mut seen, &definitional, n);
        } else {
            Self::drain_into(&mut out, &mut seen, &definitional, n);
            Self::drain_into(&mut out, &mut seen, &relational, n);
        }
        Self::drain_into(&mut out, &mut seen, &factoid, n);

        out
    }
}
