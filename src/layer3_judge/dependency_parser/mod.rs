//! Dependency parsing for improved claim extraction.
//!
//! This module provides dependency parsing capabilities for extracting
//! subject-verb-object triples and improving claim quality through
//! structural parsing.

use std::collections::HashMap;

use crate::error::JudgeError;
use crate::types::{ClaimStructure, LogicalClaim};

#[cfg(test)]
mod tests;

/// Common dependency relations in natural language.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DependencyRelation {
    /// Nominal subject (nsubj).
    NominalSubject,
    /// Direct object (dobj/obj).
    DirectObject,
    /// Indirect object (iobj).
    IndirectObject,
    /// Clausal subject (csubj).
    ClausalSubject,
    /// Passive nominal subject (nsubjpass).
    PassiveNominalSubject,
    /// Auxiliary verb (aux).
    Auxiliary,
    /// Passive auxiliary (auxpass).
    PassiveAuxiliary,
    /// Adjectival modifier (amod).
    AdjectivalModifier,
    /// Adverbial modifier (advmod).
    AdverbialModifier,
    /// Nominal modifier (nmod).
    NominalModifier,
    /// Prepositional modifier (prep).
    PrepositionalModifier,
    /// Object of preposition (pobj).
    PrepositionalObject,
    /// Determiner (det).
    Determiner,
    /// Possessive modifier (poss).
    Possessive,
    /// Compound (compound).
    Compound,
    /// Conjunct (conj).
    Conjunct,
    /// Coordinating conjunction (cc).
    CoordinatingConjunction,
    /// Marker (mark).
    Marker,
    /// Copula (cop).
    Copula,
    /// Complement clause (ccomp).
    ComplementClause,
    /// Adverbial clause (advcl).
    AdverbialClause,
    /// Relative clause modifier (relcl).
    RelativeClause,
    /// Negation modifier (neg).
    Negation,
    /// Root of the sentence.
    Root,
    /// Punctuation.
    Punctuation,
    /// Unknown relation.
    Unknown,
}

impl DependencyRelation {
    /// Parse a dependency relation from a string label.
    #[must_use]
    pub fn from_label(label: &str) -> Self {
        match label.to_lowercase().as_str() {
            "nsubj" => Self::NominalSubject,
            "dobj" | "obj" => Self::DirectObject,
            "iobj" => Self::IndirectObject,
            "csubj" => Self::ClausalSubject,
            "nsubjpass" | "nsubj:pass" => Self::PassiveNominalSubject,
            "aux" => Self::Auxiliary,
            "auxpass" | "aux:pass" => Self::PassiveAuxiliary,
            "amod" => Self::AdjectivalModifier,
            "advmod" => Self::AdverbialModifier,
            "nmod" => Self::NominalModifier,
            "prep" | "case" => Self::PrepositionalModifier,
            "pobj" => Self::PrepositionalObject,
            "det" => Self::Determiner,
            "poss" => Self::Possessive,
            "compound" => Self::Compound,
            "conj" => Self::Conjunct,
            "cc" => Self::CoordinatingConjunction,
            "mark" => Self::Marker,
            "cop" => Self::Copula,
            "ccomp" => Self::ComplementClause,
            "advcl" => Self::AdverbialClause,
            "relcl" | "acl:relcl" => Self::RelativeClause,
            "neg" => Self::Negation,
            "root" => Self::Root,
            "punct" => Self::Punctuation,
            _ => Self::Unknown,
        }
    }

    /// Convert the relation to a string label.
    #[must_use]
    pub fn as_label(&self) -> &'static str {
        match self {
            Self::NominalSubject => "nsubj",
            Self::DirectObject => "dobj",
            Self::IndirectObject => "iobj",
            Self::ClausalSubject => "csubj",
            Self::PassiveNominalSubject => "nsubjpass",
            Self::Auxiliary => "aux",
            Self::PassiveAuxiliary => "auxpass",
            Self::AdjectivalModifier => "amod",
            Self::AdverbialModifier => "advmod",
            Self::NominalModifier => "nmod",
            Self::PrepositionalModifier => "prep",
            Self::PrepositionalObject => "pobj",
            Self::Determiner => "det",
            Self::Possessive => "poss",
            Self::Compound => "compound",
            Self::Conjunct => "conj",
            Self::CoordinatingConjunction => "cc",
            Self::Marker => "mark",
            Self::Copula => "cop",
            Self::ComplementClause => "ccomp",
            Self::AdverbialClause => "advcl",
            Self::RelativeClause => "relcl",
            Self::Negation => "neg",
            Self::Root => "root",
            Self::Punctuation => "punct",
            Self::Unknown => "unknown",
        }
    }
}

/// Part-of-speech tags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PosTag {
    /// Noun.
    Noun,
    /// Proper noun.
    ProperNoun,
    /// Verb.
    Verb,
    /// Adjective.
    Adjective,
    /// Adverb.
    Adverb,
    /// Pronoun.
    Pronoun,
    /// Determiner.
    Determiner,
    /// Preposition.
    Preposition,
    /// Conjunction.
    Conjunction,
    /// Auxiliary verb.
    Auxiliary,
    /// Particle.
    Particle,
    /// Punctuation.
    Punctuation,
    /// Number.
    Number,
    /// Symbol.
    Symbol,
    /// Unknown.
    Unknown,
}

impl PosTag {
    /// Parse a POS tag from a string.
    #[must_use]
    pub fn from_tag(tag: &str) -> Self {
        let tag_upper = tag.to_uppercase();
        match tag_upper.as_str() {
            "NN" | "NNS" | "NOUN" => Self::Noun,
            "NNP" | "NNPS" | "PROPN" => Self::ProperNoun,
            "VB" | "VBD" | "VBG" | "VBN" | "VBP" | "VBZ" | "VERB" => Self::Verb,
            "JJ" | "JJR" | "JJS" | "ADJ" => Self::Adjective,
            "RB" | "RBR" | "RBS" | "ADV" => Self::Adverb,
            "PRP" | "PRP$" | "WP" | "WP$" | "PRON" => Self::Pronoun,
            "DT" | "WDT" | "PDT" | "DET" => Self::Determiner,
            "IN" | "TO" | "ADP" => Self::Preposition,
            "CC" | "CCONJ" | "SCONJ" => Self::Conjunction,
            "MD" | "AUX" => Self::Auxiliary,
            "RP" | "PART" => Self::Particle,
            "." | "," | ":" | "(" | ")" | "PUNCT" => Self::Punctuation,
            "CD" | "NUM" => Self::Number,
            "SYM" | "$" => Self::Symbol,
            _ => Self::Unknown,
        }
    }

    /// Check if this tag represents a verb.
    #[must_use]
    pub fn is_verb(&self) -> bool {
        matches!(self, Self::Verb | Self::Auxiliary)
    }

    /// Check if this tag represents a noun.
    #[must_use]
    pub fn is_noun(&self) -> bool {
        matches!(self, Self::Noun | Self::ProperNoun | Self::Pronoun)
    }

    /// Check if this tag represents a content word.
    #[must_use]
    pub fn is_content_word(&self) -> bool {
        matches!(
            self,
            Self::Noun | Self::ProperNoun | Self::Verb | Self::Adjective | Self::Adverb
        )
    }
}

/// A node in the dependency tree.
#[derive(Debug, Clone)]
pub struct DependencyNode {
    /// The token/word at this node.
    pub token: String,
    /// The lemma (base form) of the token.
    pub lemma: String,
    /// Part-of-speech tag.
    pub pos_tag: PosTag,
    /// Dependency relation to the head.
    pub relation: DependencyRelation,
    /// Index of the head node (-1 for root).
    pub head_index: i32,
    /// Index of this node in the sentence (0-indexed).
    pub index: usize,
    /// Character offset start in original text.
    pub char_start: usize,
    /// Character offset end in original text.
    pub char_end: usize,
}

impl DependencyNode {
    /// Create a new dependency node.
    #[must_use]
    pub fn new(token: &str, index: usize) -> Self {
        Self {
            token: token.to_string(),
            lemma: token.to_lowercase(),
            pos_tag: PosTag::Unknown,
            relation: DependencyRelation::Unknown,
            head_index: -1,
            index,
            char_start: 0,
            char_end: 0,
        }
    }

    /// Set the lemma.
    #[must_use]
    pub fn with_lemma(mut self, lemma: &str) -> Self {
        self.lemma = lemma.to_string();
        self
    }

    /// Set the POS tag.
    #[must_use]
    pub fn with_pos(mut self, pos: PosTag) -> Self {
        self.pos_tag = pos;
        self
    }

    /// Set the dependency relation.
    #[must_use]
    pub fn with_relation(mut self, relation: DependencyRelation) -> Self {
        self.relation = relation;
        self
    }

    /// Set the head index.
    #[must_use]
    pub fn with_head(mut self, head_index: i32) -> Self {
        self.head_index = head_index;
        self
    }

    /// Set the character offsets.
    #[must_use]
    pub fn with_offsets(mut self, start: usize, end: usize) -> Self {
        self.char_start = start;
        self.char_end = end;
        self
    }

    /// Check if this node is the root.
    #[must_use]
    pub fn is_root(&self) -> bool {
        self.head_index < 0 || self.relation == DependencyRelation::Root
    }
}

/// A dependency tree representing the syntactic structure of a sentence.
#[derive(Debug, Clone)]
pub struct DependencyTree {
    /// The nodes in the tree (ordered by position in sentence).
    pub nodes: Vec<DependencyNode>,
    /// The original sentence text.
    pub text: String,
    /// Index of the root node.
    pub root_index: Option<usize>,
}

impl DependencyTree {
    /// Create a new empty dependency tree.
    #[must_use]
    pub fn new(text: &str) -> Self {
        Self {
            nodes: Vec::new(),
            text: text.to_string(),
            root_index: None,
        }
    }

    /// Add a node to the tree.
    pub fn add_node(&mut self, node: DependencyNode) {
        if node.is_root() && self.root_index.is_none() {
            self.root_index = Some(node.index);
        }
        self.nodes.push(node);
    }

    /// Get a node by index.
    #[must_use]
    pub fn get_node(&self, index: usize) -> Option<&DependencyNode> {
        self.nodes.iter().find(|n| n.index == index)
    }

    /// Get the root node.
    #[must_use]
    pub fn root(&self) -> Option<&DependencyNode> {
        self.root_index.and_then(|i| self.get_node(i))
    }

    /// Get all children of a node.
    #[must_use]
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    pub fn children(&self, index: usize) -> Vec<&DependencyNode> {
        self.nodes
            .iter()
            .filter(|n| n.head_index == index as i32)
            .collect()
    }

    /// Get children with a specific relation.
    #[must_use]
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    pub fn children_with_relation(
        &self,
        index: usize,
        relation: DependencyRelation,
    ) -> Vec<&DependencyNode> {
        self.nodes
            .iter()
            .filter(|n| n.head_index == index as i32 && n.relation == relation)
            .collect()
    }

    /// Find nodes with a specific relation.
    #[must_use]
    pub fn find_by_relation(&self, relation: DependencyRelation) -> Vec<&DependencyNode> {
        self.nodes
            .iter()
            .filter(|n| n.relation == relation)
            .collect()
    }

    /// Find nodes with a specific POS tag.
    #[must_use]
    pub fn find_by_pos(&self, pos: PosTag) -> Vec<&DependencyNode> {
        self.nodes.iter().filter(|n| n.pos_tag == pos).collect()
    }

    /// Get the subtree text for a node (including all descendants).
    #[must_use]
    pub fn subtree_text(&self, index: usize) -> String {
        let mut indices = vec![index];
        let mut i = 0;
        while i < indices.len() {
            let current = indices[i];
            for child in self.children(current) {
                indices.push(child.index);
            }
            i += 1;
        }
        indices.sort_unstable();

        indices
            .iter()
            .filter_map(|&idx| self.get_node(idx))
            .map(|n| n.token.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Check if the tree is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get the number of nodes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
}

/// A subject-verb-object triple extracted from a sentence.
#[derive(Debug, Clone, PartialEq)]
pub struct SvoTriple {
    /// The subject of the triple.
    pub subject: String,
    /// The verb/predicate of the triple.
    pub verb: String,
    /// The object of the triple (if present).
    pub object: Option<String>,
    /// Whether this triple is negated.
    pub negated: bool,
    /// Confidence in this extraction.
    pub confidence: f32,
}

impl SvoTriple {
    /// Create a new SVO triple.
    #[must_use]
    pub fn new(subject: &str, verb: &str, object: Option<&str>) -> Self {
        Self {
            subject: subject.to_string(),
            verb: verb.to_string(),
            object: object.map(ToString::to_string),
            negated: false,
            confidence: 1.0,
        }
    }

    /// Mark this triple as negated.
    #[must_use]
    pub fn with_negation(mut self, negated: bool) -> Self {
        self.negated = negated;
        self
    }

    /// Set the confidence.
    #[must_use]
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }
}

/// Trait for dependency parsing.
pub trait DependencyParser: Send + Sync {
    /// Parse a text into a dependency tree.
    fn parse(&self, text: &str) -> DependencyTree;

    /// Extract subject-verb-object triples from a dependency tree.
    fn extract_subject_verb_object(&self, tree: &DependencyTree) -> Vec<SvoTriple>;
}

/// A simple pattern-based dependency parser.
///
/// This parser uses heuristics and patterns to approximate dependency parsing
/// without requiring external NLP libraries.
pub struct SimpleDependencyParser {
    /// Common verbs to identify.
    verbs: HashMap<String, String>,
    /// Common auxiliaries.
    auxiliaries: Vec<String>,
    /// Common determiners.
    determiners: Vec<String>,
    /// Common prepositions.
    prepositions: Vec<String>,
    /// Negation words.
    negations: Vec<String>,
}

impl Default for SimpleDependencyParser {
    fn default() -> Self {
        Self::new()
    }
}

impl SimpleDependencyParser {
    /// Create a new simple dependency parser.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn new() -> Self {
        let mut verbs = HashMap::new();
        // Be verbs
        for (form, lemma) in [
            ("is", "be"),
            ("are", "be"),
            ("was", "be"),
            ("were", "be"),
            ("am", "be"),
            ("been", "be"),
            ("being", "be"),
        ] {
            verbs.insert(form.to_string(), lemma.to_string());
        }
        // Common verbs
        for (form, lemma) in [
            ("has", "have"),
            ("have", "have"),
            ("had", "have"),
            ("having", "have"),
            ("does", "do"),
            ("do", "do"),
            ("did", "do"),
            ("makes", "make"),
            ("make", "make"),
            ("made", "make"),
            ("causes", "cause"),
            ("cause", "cause"),
            ("caused", "cause"),
            ("shows", "show"),
            ("show", "show"),
            ("showed", "show"),
            ("contains", "contain"),
            ("contain", "contain"),
            ("contained", "contain"),
            ("includes", "include"),
            ("include", "include"),
            ("included", "include"),
            ("becomes", "become"),
            ("become", "become"),
            ("became", "become"),
            ("provides", "provide"),
            ("provide", "provide"),
            ("provided", "provide"),
            ("requires", "require"),
            ("require", "require"),
            ("required", "require"),
            ("uses", "use"),
            ("use", "use"),
            ("used", "use"),
            ("creates", "create"),
            ("create", "create"),
            ("created", "create"),
            ("supports", "support"),
            ("support", "support"),
            ("supported", "support"),
            ("enables", "enable"),
            ("enable", "enable"),
            ("enabled", "enable"),
            ("allows", "allow"),
            ("allow", "allow"),
            ("allowed", "allow"),
            ("produces", "produce"),
            ("produce", "produce"),
            ("produced", "produce"),
            ("generates", "generate"),
            ("generate", "generate"),
            ("generated", "generate"),
            ("leads", "lead"),
            ("lead", "lead"),
            ("led", "lead"),
            ("results", "result"),
            ("result", "result"),
            ("resulted", "result"),
            ("affects", "affect"),
            ("affect", "affect"),
            ("affected", "affect"),
            ("influences", "influence"),
            ("influence", "influence"),
            ("influenced", "influence"),
            ("determines", "determine"),
            ("determine", "determine"),
            ("determined", "determine"),
            ("represents", "represent"),
            ("represent", "represent"),
            ("represented", "represent"),
            ("consists", "consist"),
            ("consist", "consist"),
            ("consisted", "consist"),
            ("exists", "exist"),
            ("exist", "exist"),
            ("existed", "exist"),
            ("occurs", "occur"),
            ("occur", "occur"),
            ("occurred", "occur"),
            ("improves", "improve"),
            ("improve", "improve"),
            ("improved", "improve"),
            ("reducing", "reduce"),
            ("reduces", "reduce"),
            ("reduce", "reduce"),
            ("reduced", "reduce"),
            ("increases", "increase"),
            ("increase", "increase"),
            ("increased", "increase"),
            ("decreases", "decrease"),
            ("decrease", "decrease"),
            ("decreased", "decrease"),
            ("changes", "change"),
            ("change", "change"),
            ("changed", "change"),
            ("gives", "give"),
            ("give", "give"),
            ("gave", "give"),
            ("given", "give"),
            ("takes", "take"),
            ("take", "take"),
            ("took", "take"),
            ("taken", "take"),
            ("gets", "get"),
            ("get", "get"),
            ("got", "get"),
            ("gotten", "get"),
            ("knows", "know"),
            ("know", "know"),
            ("knew", "know"),
            ("known", "know"),
            ("thinks", "think"),
            ("think", "think"),
            ("thought", "think"),
            ("believes", "believe"),
            ("believe", "believe"),
            ("believed", "believe"),
            ("says", "say"),
            ("say", "say"),
            ("said", "say"),
            ("tells", "tell"),
            ("tell", "tell"),
            ("told", "tell"),
            ("sees", "see"),
            ("see", "see"),
            ("saw", "see"),
            ("seen", "see"),
            ("finds", "find"),
            ("find", "find"),
            ("found", "find"),
            ("wants", "want"),
            ("want", "want"),
            ("wanted", "want"),
            ("needs", "need"),
            ("need", "need"),
            ("needed", "need"),
            ("seems", "seem"),
            ("seem", "seem"),
            ("seemed", "seem"),
            ("appears", "appear"),
            ("appear", "appear"),
            ("appeared", "appear"),
            ("works", "work"),
            ("work", "work"),
            ("worked", "work"),
            ("moves", "move"),
            ("move", "move"),
            ("moved", "move"),
            ("comes", "come"),
            ("come", "come"),
            ("came", "come"),
            ("goes", "go"),
            ("go", "go"),
            ("went", "go"),
            ("gone", "go"),
            ("runs", "run"),
            ("run", "run"),
            ("ran", "run"),
            ("helps", "help"),
            ("help", "help"),
            ("helped", "help"),
            ("starts", "start"),
            ("start", "start"),
            ("started", "start"),
            ("stops", "stop"),
            ("stop", "stop"),
            ("stopped", "stop"),
            ("keeps", "keep"),
            ("keep", "keep"),
            ("kept", "keep"),
            ("brings", "bring"),
            ("bring", "bring"),
            ("brought", "bring"),
            ("holds", "hold"),
            ("hold", "hold"),
            ("held", "hold"),
            ("means", "mean"),
            ("mean", "mean"),
            ("meant", "mean"),
            ("shows", "show"),
            ("show", "show"),
            ("showed", "show"),
            ("shown", "show"),
            ("plays", "play"),
            ("play", "play"),
            ("played", "play"),
            ("reads", "read"),
            ("read", "read"),
            ("writes", "write"),
            ("write", "write"),
            ("wrote", "write"),
            ("written", "write"),
            ("lives", "live"),
            ("live", "live"),
            ("lived", "live"),
            ("dies", "die"),
            ("die", "die"),
            ("died", "die"),
        ] {
            verbs.insert(form.to_string(), lemma.to_string());
        }

        Self {
            verbs,
            auxiliaries: vec![
                "can".to_string(),
                "could".to_string(),
                "will".to_string(),
                "would".to_string(),
                "shall".to_string(),
                "should".to_string(),
                "may".to_string(),
                "might".to_string(),
                "must".to_string(),
            ],
            determiners: vec![
                "the".to_string(),
                "a".to_string(),
                "an".to_string(),
                "this".to_string(),
                "that".to_string(),
                "these".to_string(),
                "those".to_string(),
                "my".to_string(),
                "your".to_string(),
                "his".to_string(),
                "her".to_string(),
                "its".to_string(),
                "our".to_string(),
                "their".to_string(),
                "some".to_string(),
                "any".to_string(),
                "no".to_string(),
                "every".to_string(),
                "each".to_string(),
                "all".to_string(),
            ],
            prepositions: vec![
                "in".to_string(),
                "on".to_string(),
                "at".to_string(),
                "to".to_string(),
                "for".to_string(),
                "with".to_string(),
                "by".to_string(),
                "from".to_string(),
                "of".to_string(),
                "about".to_string(),
                "into".to_string(),
                "through".to_string(),
                "during".to_string(),
                "before".to_string(),
                "after".to_string(),
                "above".to_string(),
                "below".to_string(),
                "between".to_string(),
                "under".to_string(),
                "over".to_string(),
            ],
            negations: vec![
                "not".to_string(),
                "n't".to_string(),
                "never".to_string(),
                "no".to_string(),
                "neither".to_string(),
                "nor".to_string(),
            ],
        }
    }

    /// Tokenize text into words with positions.
    #[allow(clippy::unused_self)]
    fn tokenize(&self, text: &str) -> Vec<(String, usize, usize)> {
        let mut tokens = Vec::new();
        let mut start = 0;
        let mut in_word = false;
        let mut word_start = 0;

        for (i, c) in text.char_indices() {
            if c.is_alphanumeric() || c == '\'' || c == '-' {
                if !in_word {
                    word_start = i;
                    in_word = true;
                }
            } else {
                if in_word {
                    let word = &text[word_start..i];
                    if !word.is_empty() {
                        tokens.push((word.to_string(), word_start, i));
                    }
                    in_word = false;
                }
                // Handle punctuation as separate tokens
                if c.is_ascii_punctuation() && !c.is_whitespace() {
                    tokens.push((c.to_string(), i, i + 1));
                }
            }
            start = i + c.len_utf8();
        }

        // Handle last word
        if in_word {
            let word = &text[word_start..];
            if !word.is_empty() {
                tokens.push((word.to_string(), word_start, start));
            }
        }

        tokens
    }

    /// Determine the POS tag for a token.
    fn get_pos_tag(&self, token: &str, prev_token: Option<&str>) -> PosTag {
        let lower = token.to_lowercase();

        // Check for punctuation
        if token.len() == 1
            && token
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_punctuation())
        {
            return PosTag::Punctuation;
        }

        // Check for determiners
        if self.determiners.contains(&lower) {
            return PosTag::Determiner;
        }

        // Check for auxiliaries
        if self.auxiliaries.contains(&lower) {
            return PosTag::Auxiliary;
        }

        // Check for verbs
        if self.verbs.contains_key(&lower) {
            return PosTag::Verb;
        }

        // Check for prepositions
        if self.prepositions.contains(&lower) {
            return PosTag::Preposition;
        }

        // Check for conjunctions
        if lower == "and" || lower == "or" || lower == "but" {
            return PosTag::Conjunction;
        }

        // Check for pronouns
        if matches!(
            lower.as_str(),
            "i" | "you"
                | "he"
                | "she"
                | "it"
                | "we"
                | "they"
                | "me"
                | "him"
                | "her"
                | "us"
                | "them"
        ) {
            return PosTag::Pronoun;
        }

        // Check for common adverbs
        if lower.ends_with("ly")
            || matches!(
                lower.as_str(),
                "very" | "also" | "too" | "always" | "often" | "never" | "just" | "only" | "still"
            )
        {
            return PosTag::Adverb;
        }

        // Check for numbers
        if token.chars().all(|c| c.is_ascii_digit()) {
            return PosTag::Number;
        }

        // Heuristic: words following determiners are likely nouns
        if let Some(prev) = prev_token
            && self.determiners.contains(&prev.to_lowercase())
        {
            return PosTag::Noun;
        }

        // Heuristic: words ending in common noun suffixes
        if lower.ends_with("tion")
            || lower.ends_with("ness")
            || lower.ends_with("ment")
            || lower.ends_with("ity")
            || lower.ends_with("ence")
            || lower.ends_with("ance")
        {
            return PosTag::Noun;
        }

        // Heuristic: words ending in common adjective suffixes
        if lower.ends_with("ful")
            || lower.ends_with("less")
            || lower.ends_with("ous")
            || lower.ends_with("ive")
            || lower.ends_with("able")
            || lower.ends_with("ible")
        {
            return PosTag::Adjective;
        }

        // Heuristic: capitalized words are often proper nouns
        if token.chars().next().is_some_and(char::is_uppercase) {
            return PosTag::ProperNoun;
        }

        // Default to noun
        PosTag::Noun
    }

    /// Get the lemma for a token.
    fn get_lemma(&self, token: &str) -> String {
        let lower = token.to_lowercase();
        if let Some(lemma) = self.verbs.get(&lower) {
            return lemma.clone();
        }
        lower
    }

    /// Find the main verb in a token sequence.
    pub(crate) fn find_main_verb(&self, tokens: &[(String, usize, usize)]) -> Option<usize> {
        // First pass: look for known verbs
        for (i, (token, _, _)) in tokens.iter().enumerate() {
            let lower = token.to_lowercase();
            if self.verbs.contains_key(&lower) && !self.auxiliaries.contains(&lower) {
                return Some(i);
            }
        }

        // Second pass: look for auxiliaries if no main verb found
        for (i, (token, _, _)) in tokens.iter().enumerate() {
            let lower = token.to_lowercase();
            if self.auxiliaries.contains(&lower) || self.verbs.contains_key(&lower) {
                return Some(i);
            }
        }

        None
    }

    /// Check if there's a negation before an index.
    pub(crate) fn has_negation_before(
        &self,
        tokens: &[(String, usize, usize)],
        index: usize,
    ) -> bool {
        for i in (0..index).rev() {
            let lower = tokens[i].0.to_lowercase();
            if self.negations.contains(&lower) {
                return true;
            }
            // Don't look too far back
            if index - i > 3 {
                break;
            }
        }
        false
    }

    /// Extract noun phrase starting from an index.
    pub(crate) fn extract_noun_phrase(
        &self,
        tokens: &[(String, usize, usize)],
        start: usize,
    ) -> String {
        let mut parts = Vec::new();
        let mut i = start;

        while i < tokens.len() {
            let (token, _, _) = &tokens[i];
            let lower = token.to_lowercase();

            // Stop at verbs, prepositions (except 'of'), punctuation
            if self.verbs.contains_key(&lower)
                || self.auxiliaries.contains(&lower)
                || (self.prepositions.contains(&lower) && lower != "of")
                || token.len() == 1
                    && token
                        .chars()
                        .next()
                        .is_some_and(|c| c.is_ascii_punctuation())
            {
                break;
            }

            // Skip determiners in the middle
            if i > start && self.determiners.contains(&lower) {
                i += 1;
                continue;
            }

            parts.push(token.as_str());
            i += 1;
        }

        parts.join(" ")
    }

    /// Extract noun phrase going backwards from an index.
    pub(crate) fn extract_noun_phrase_backward(
        &self,
        tokens: &[(String, usize, usize)],
        end: usize,
    ) -> String {
        let mut parts = Vec::new();
        let mut i = end;

        loop {
            let (token, _, _) = &tokens[i];
            let lower = token.to_lowercase();

            // Stop at verbs, some prepositions, punctuation
            if self.verbs.contains_key(&lower)
                || self.auxiliaries.contains(&lower)
                || (self.prepositions.contains(&lower) && lower != "of")
                || (token.len() == 1
                    && token
                        .chars()
                        .next()
                        .is_some_and(|c| c.is_ascii_punctuation()))
            {
                break;
            }

            parts.push(token.as_str());

            if i == 0 {
                break;
            }
            i -= 1;
        }

        parts.reverse();
        parts.join(" ")
    }
}

impl DependencyParser for SimpleDependencyParser {
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    fn parse(&self, text: &str) -> DependencyTree {
        let mut tree = DependencyTree::new(text);
        let tokens = self.tokenize(text);

        if tokens.is_empty() {
            return tree;
        }

        // Find the main verb (will be the root)
        let main_verb_idx = self.find_main_verb(&tokens);

        let mut prev_token: Option<&str> = None;

        for (i, (token, start, end)) in tokens.iter().enumerate() {
            let pos = self.get_pos_tag(token, prev_token);
            let lemma = self.get_lemma(token);

            let (relation, head) = if Some(i) == main_verb_idx {
                (DependencyRelation::Root, -1)
            } else if let Some(verb_idx) = main_verb_idx {
                // Simple heuristics for relations
                let lower = token.to_lowercase();

                if i < verb_idx {
                    // Before the verb
                    if self.determiners.contains(&lower) {
                        (DependencyRelation::Determiner, (i + 1) as i32)
                    } else if self.auxiliaries.contains(&lower) {
                        (DependencyRelation::Auxiliary, verb_idx as i32)
                    } else if self.negations.contains(&lower) {
                        (DependencyRelation::Negation, verb_idx as i32)
                    } else if pos.is_noun() || pos == PosTag::ProperNoun {
                        (DependencyRelation::NominalSubject, verb_idx as i32)
                    } else if pos == PosTag::Adjective {
                        (DependencyRelation::AdjectivalModifier, (i + 1) as i32)
                    } else {
                        (DependencyRelation::Unknown, verb_idx as i32)
                    }
                } else {
                    // After the verb
                    if self.determiners.contains(&lower) {
                        (
                            DependencyRelation::Determiner,
                            (i + 1).min(tokens.len() - 1) as i32,
                        )
                    } else if self.prepositions.contains(&lower) {
                        (DependencyRelation::PrepositionalModifier, verb_idx as i32)
                    } else if self.negations.contains(&lower) {
                        (DependencyRelation::Negation, verb_idx as i32)
                    } else if pos.is_noun() || pos == PosTag::ProperNoun {
                        // Check if preceded by preposition
                        if i > 0 && self.prepositions.contains(&tokens[i - 1].0.to_lowercase()) {
                            (DependencyRelation::PrepositionalObject, (i - 1) as i32)
                        } else {
                            (DependencyRelation::DirectObject, verb_idx as i32)
                        }
                    } else if pos == PosTag::Adjective {
                        (DependencyRelation::AdjectivalModifier, verb_idx as i32)
                    } else if pos == PosTag::Adverb {
                        (DependencyRelation::AdverbialModifier, verb_idx as i32)
                    } else if pos == PosTag::Punctuation {
                        (DependencyRelation::Punctuation, verb_idx as i32)
                    } else {
                        (DependencyRelation::Unknown, verb_idx as i32)
                    }
                }
            } else {
                // No verb found - attach to previous or next
                if i > 0 {
                    (DependencyRelation::Unknown, (i - 1) as i32)
                } else {
                    (DependencyRelation::Root, -1)
                }
            };

            let node = DependencyNode::new(token, i)
                .with_lemma(&lemma)
                .with_pos(pos)
                .with_relation(relation)
                .with_head(head)
                .with_offsets(*start, *end);

            tree.add_node(node);
            prev_token = Some(token);
        }

        tree
    }

    fn extract_subject_verb_object(&self, tree: &DependencyTree) -> Vec<SvoTriple> {
        let mut triples = Vec::new();

        // Find the root (main verb)
        let root = match tree.root() {
            Some(r) if r.pos_tag.is_verb() || r.pos_tag == PosTag::Auxiliary => r,
            _ => return triples,
        };

        // Find subjects
        let subjects = tree.children_with_relation(root.index, DependencyRelation::NominalSubject);

        // Find objects
        let objects = tree.children_with_relation(root.index, DependencyRelation::DirectObject);

        // Check for negation
        let negations = tree.children_with_relation(root.index, DependencyRelation::Negation);
        let is_negated = !negations.is_empty();

        // Build triples
        for subj in &subjects {
            let subject_text = tree.subtree_text(subj.index);

            if objects.is_empty() {
                // Handle copula constructions (e.g., "X is Y")
                // Look for adjectives or noun complements
                let adj_mods =
                    tree.children_with_relation(root.index, DependencyRelation::AdjectivalModifier);
                let prep_mods = tree
                    .children_with_relation(root.index, DependencyRelation::PrepositionalModifier);

                if !adj_mods.is_empty() {
                    for adj in adj_mods {
                        let triple = SvoTriple::new(&subject_text, &root.lemma, Some(&adj.token))
                            .with_negation(is_negated)
                            .with_confidence(0.8);
                        triples.push(triple);
                    }
                } else if !prep_mods.is_empty() {
                    for prep in prep_mods {
                        let prep_obj = tree.subtree_text(prep.index);
                        let triple = SvoTriple::new(&subject_text, &root.lemma, Some(&prep_obj))
                            .with_negation(is_negated)
                            .with_confidence(0.7);
                        triples.push(triple);
                    }
                } else {
                    // Subject-verb only
                    let triple = SvoTriple::new(&subject_text, &root.lemma, None)
                        .with_negation(is_negated)
                        .with_confidence(0.6);
                    triples.push(triple);
                }
            } else {
                for obj in &objects {
                    let object_text = tree.subtree_text(obj.index);
                    let triple = SvoTriple::new(&subject_text, &root.lemma, Some(&object_text))
                        .with_negation(is_negated)
                        .with_confidence(0.9);
                    triples.push(triple);
                }
            }
        }

        // If no subjects found with proper relations, try heuristic extraction
        if triples.is_empty() && !tree.nodes.is_empty() {
            let tokens: Vec<_> = tree
                .nodes
                .iter()
                .map(|n| (n.token.clone(), n.char_start, n.char_end))
                .collect();
            if let Some(verb_idx) = self.find_main_verb(&tokens) {
                let is_neg = self.has_negation_before(&tokens, verb_idx);

                // Extract subject (before verb)
                let subject = if verb_idx > 0 {
                    self.extract_noun_phrase_backward(&tokens, verb_idx - 1)
                } else {
                    String::new()
                };

                // Extract object (after verb)
                let object = if verb_idx + 1 < tokens.len() {
                    let obj = self.extract_noun_phrase(&tokens, verb_idx + 1);
                    if obj.is_empty() { None } else { Some(obj) }
                } else {
                    None
                };

                if !subject.is_empty() {
                    let verb_lemma = self.get_lemma(&tokens[verb_idx].0);
                    let triple = SvoTriple::new(&subject, &verb_lemma, object.as_deref())
                        .with_negation(is_neg)
                        .with_confidence(0.5);
                    triples.push(triple);
                }
            }
        }

        triples
    }
}

/// A claim extractor that uses dependency parsing for improved extraction.
pub struct DependencyClaimExtractor {
    /// The dependency parser to use.
    parser: Box<dyn DependencyParser>,
}

impl Default for DependencyClaimExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl DependencyClaimExtractor {
    /// Create a new dependency claim extractor with the default parser.
    #[must_use]
    pub fn new() -> Self {
        Self {
            parser: Box::new(SimpleDependencyParser::new()),
        }
    }

    /// Create a new dependency claim extractor with a custom parser.
    #[must_use]
    pub fn with_parser(parser: Box<dyn DependencyParser>) -> Self {
        Self { parser }
    }

    /// Split text into sentences.
    fn split_sentences(text: &str) -> Vec<String> {
        text.split(['.', '!', '?'])
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty() && s.split_whitespace().count() >= 2)
            .collect()
    }

    /// Extract claims from a single sentence.
    fn extract_from_sentence(&self, sentence: &str) -> Vec<LogicalClaim> {
        let tree = self.parser.parse(sentence);
        let triples = self.parser.extract_subject_verb_object(&tree);

        triples
            .into_iter()
            .map(|triple| {
                let structure = if triple.negated {
                    ClaimStructure::Not(Box::new(ClaimStructure::Predicate {
                        subject: triple.subject.clone(),
                        predicate: triple.verb.clone(),
                        object: triple.object.clone(),
                    }))
                } else {
                    ClaimStructure::Predicate {
                        subject: triple.subject,
                        predicate: triple.verb,
                        object: triple.object,
                    }
                };

                LogicalClaim::new(sentence, structure).with_confidence(triple.confidence)
            })
            .collect()
    }

    /// Extract all claims from text.
    ///
    /// # Errors
    ///
    /// Returns an error if claim extraction fails.
    pub fn extract_claims(
        &self,
        text: &str,
        max_claims: usize,
    ) -> Result<Vec<LogicalClaim>, JudgeError> {
        let sentences = Self::split_sentences(text);

        let claims: Vec<LogicalClaim> = sentences
            .into_iter()
            .flat_map(|s| self.extract_from_sentence(&s))
            .take(max_claims)
            .collect();

        Ok(claims)
    }

    /// Extract relationships from text as SVO triples.
    #[must_use]
    pub fn extract_relationships(&self, text: &str) -> Vec<SvoTriple> {
        let sentences = Self::split_sentences(text);

        sentences
            .into_iter()
            .flat_map(|s| {
                let tree = self.parser.parse(&s);
                self.parser.extract_subject_verb_object(&tree)
            })
            .collect()
    }

    /// Parse text into dependency trees.
    #[must_use]
    pub fn parse_sentences(&self, text: &str) -> Vec<DependencyTree> {
        let sentences = Self::split_sentences(text);
        sentences
            .into_iter()
            .map(|s| self.parser.parse(&s))
            .collect()
    }
}
