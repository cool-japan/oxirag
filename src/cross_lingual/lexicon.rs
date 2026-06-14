//! Bilingual lexicon and shared-form text normalization.

use std::collections::HashMap;

// ── Diacritic normalization ─────────────────────────────────────────────────

/// Map a single character to its diacritic-stripped ASCII equivalent.
///
/// Covers the common Latin-script accents and ligatures encountered across
/// Western European languages (French, German, Spanish, Portuguese, Italian,
/// the Nordic languages, and Polish). Characters without a known mapping are
/// returned unchanged.
fn strip_diacritic(c: char) -> char {
    match c {
        'á' | 'à' | 'â' | 'ä' | 'ã' | 'å' | 'ā' | 'ą' => 'a',
        'ç' | 'ć' | 'č' => 'c',
        'é' | 'è' | 'ê' | 'ë' | 'ē' | 'ę' | 'ě' => 'e',
        'í' | 'ì' | 'î' | 'ï' | 'ī' => 'i',
        'ñ' | 'ń' => 'n',
        'ó' | 'ò' | 'ô' | 'ö' | 'õ' | 'ø' | 'ō' => 'o',
        'ú' | 'ù' | 'û' | 'ü' | 'ū' | 'ů' => 'u',
        'ý' | 'ÿ' => 'y',
        'ś' | 'š' => 's',
        'ł' => 'l',
        'ź' | 'ż' | 'ž' => 'z',
        other => other,
    }
}

/// Lowercase `text` and strip common Latin diacritics to a shared ASCII form.
///
/// This collapses accented and unaccented spellings (for example `café` and
/// `cafe`, or `München` and `munchen`) onto the same surface form so that a
/// query in one orthography can match a document in another. The German
/// eszett `ß` is transliterated to `ss`.
#[must_use]
pub fn normalize(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for lowered in text.chars().flat_map(char::to_lowercase) {
        if lowered == 'ß' {
            out.push_str("ss");
        } else {
            out.push(strip_diacritic(lowered));
        }
    }
    out
}

// ── BilingualLexicon ────────────────────────────────────────────────────────

/// A caller-supplied dictionary mapping source-language tokens to one or more
/// target-language translations.
///
/// The lexicon drives cross-lingual query expansion: each query token is looked
/// up and its translations are appended to the search terms, so a query phrased
/// in the source language can match documents written in the target language.
/// All keys and values are stored lowercased.
#[derive(Debug, Clone, Default)]
pub struct BilingualLexicon {
    /// Source token → list of target translations (insertion-ordered, deduped).
    map: HashMap<String, Vec<String>>,
}

impl BilingualLexicon {
    /// Create a new, empty lexicon.
    #[must_use]
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    /// Add a `source` → `target` translation pair.
    ///
    /// Both ends are lowercased before storage. Adding the same target for an
    /// existing source is idempotent: duplicate translations are not appended.
    pub fn add(&mut self, source: &str, target: &str) {
        let source = source.to_lowercase();
        let target = target.to_lowercase();
        let entry = self.map.entry(source).or_default();
        if !entry.contains(&target) {
            entry.push(target);
        }
    }

    /// Return the translations registered for `token`.
    ///
    /// The lookup is case-insensitive. An unknown token yields an empty vector.
    #[must_use]
    pub fn translate(&self, token: &str) -> Vec<String> {
        self.map
            .get(&token.to_lowercase())
            .cloned()
            .unwrap_or_default()
    }

    /// Number of distinct source tokens held by the lexicon.
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Return `true` when the lexicon holds no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}
