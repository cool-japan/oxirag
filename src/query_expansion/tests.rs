//! Tests for query expansion strategies.

#[cfg(all(test, not(target_arch = "wasm32")))]
#[allow(clippy::similar_names)]
mod inner {
    use std::collections::HashMap;
    use std::collections::HashSet;

    use crate::query_expansion::composite::CompositeExpander;
    use crate::query_expansion::ngram::NGramExpander;
    use crate::query_expansion::prf::PseudoRelevanceFeedback;
    use crate::query_expansion::reformulator::QueryReformulator;
    use crate::query_expansion::stem::StemExpander;
    use crate::query_expansion::synonym::SynonymExpander;
    use crate::query_expansion::types::{
        ExpandedQuery, ExpansionConfig, ExpansionMethod, QueryExpander,
    };
    use crate::types::{Document, Query, SearchResult};

    fn create_test_query(text: &str) -> Query {
        Query::new(text)
    }

    fn create_test_results() -> Vec<SearchResult> {
        vec![
            SearchResult::new(
                Document::new("Machine learning algorithms process large datasets efficiently"),
                0.9,
                0,
            ),
            SearchResult::new(
                Document::new("Deep learning neural networks achieve remarkable results"),
                0.85,
                1,
            ),
            SearchResult::new(
                Document::new("Training machine learning models requires significant computation"),
                0.8,
                2,
            ),
        ]
    }

    #[test]
    fn test_expansion_method_display() {
        assert_eq!(format!("{}", ExpansionMethod::Synonyms), "synonyms");
        assert_eq!(format!("{}", ExpansionMethod::Stemming), "stemming");
        assert_eq!(format!("{}", ExpansionMethod::NGrams), "ngrams");
    }

    #[test]
    fn test_expanded_query_creation() {
        let query = create_test_query("machine learning");
        let expanded = ExpandedQuery::new(query.clone(), ExpansionMethod::Synonyms)
            .with_term("computer")
            .with_term("training")
            .with_weight(0.8);

        assert_eq!(expanded.original_query.text, "machine learning");
        assert_eq!(expanded.expanded_terms.len(), 2);
        assert!((expanded.weight - 0.8).abs() < f32::EPSILON);
        assert_eq!(expanded.expansion_method, ExpansionMethod::Synonyms);
    }

    #[test]
    fn test_expanded_query_to_combined() {
        let query = create_test_query("search");
        let expanded =
            ExpandedQuery::new(query, ExpansionMethod::Synonyms).with_terms(vec!["find", "lookup"]);

        let combined = expanded.to_combined_query();
        assert!(combined.contains("search"));
        assert!(combined.contains("find"));
        assert!(combined.contains("lookup"));
    }

    #[test]
    fn test_expansion_config_default() {
        let config = ExpansionConfig::default();
        assert_eq!(config.max_expansions, 5);
        assert!((config.synonym_weight - 0.8).abs() < f32::EPSILON);
        assert_eq!(config.ngram_range, (2, 3));
        assert_eq!(config.prf_documents, 3);
    }

    #[test]
    fn test_expansion_config_builder() {
        let config = ExpansionConfig::new()
            .with_max_expansions(10)
            .with_synonym_weight(0.9)
            .with_ngram_range(2, 4)
            .with_prf_documents(5);

        assert_eq!(config.max_expansions, 10);
        assert!((config.synonym_weight - 0.9).abs() < f32::EPSILON);
        assert_eq!(config.ngram_range, (2, 4));
        assert_eq!(config.prf_documents, 5);
    }

    #[test]
    fn test_synonym_weight_clamping() {
        let config = ExpansionConfig::new().with_synonym_weight(1.5);
        assert!((config.synonym_weight - 1.0).abs() < f32::EPSILON);

        let config = ExpansionConfig::new().with_synonym_weight(-0.5);
        assert!(config.synonym_weight.abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn test_synonym_expander() {
        let expander = SynonymExpander::default();
        let query = create_test_query("search data");

        let expanded = expander.expand(&query).await;
        assert!(!expanded.is_empty());

        // Check that synonyms are added
        let combined_text = expanded
            .iter()
            .map(|q| q.text.clone())
            .collect::<Vec<_>>()
            .join(" ");

        assert!(combined_text.contains("search"));
        // Should have synonyms for "search" and/or "data"
        let has_synonym = combined_text.contains("find")
            || combined_text.contains("query")
            || combined_text.contains("information");
        assert!(has_synonym);
    }

    #[tokio::test]
    async fn test_synonym_expander_get_synonyms() {
        let expander = SynonymExpander::default();

        let synonyms = expander.get_synonyms("search");
        assert!(synonyms.contains(&"find".to_string()));

        let synonyms = expander.get_synonyms("SEARCH");
        assert!(synonyms.contains(&"find".to_string()));

        let synonyms = expander.get_synonyms("nonexistent");
        assert!(synonyms.is_empty());
    }

    #[tokio::test]
    async fn test_synonym_expander_reformulate() {
        // Use a custom config with lower min_frequency to ensure expansion
        let config = ExpansionConfig {
            prf_min_frequency: 1,
            ..Default::default()
        };
        let expander = SynonymExpander::new(config);
        let query = create_test_query("machine learning");
        let results = create_test_results();

        let reformulated = expander.reformulate(&query, &results).await;
        assert!(reformulated.text.contains("machine learning"));
        // Should have added terms from results (now that min_frequency is 1)
        assert!(reformulated.text.len() > query.text.len());
    }

    #[test]
    fn test_stem_expander_stem() {
        let expander = StemExpander::default();

        assert_eq!(expander.stem("running"), "runn");
        assert_eq!(expander.stem("algorithms"), "algorithm");
        assert_eq!(expander.stem("processing"), "process");
    }

    #[test]
    fn test_stem_expander_word_forms() {
        let expander = StemExpander::default();
        let forms = expander.get_word_forms("learn");

        assert!(forms.contains(&"learns".to_string()));
        assert!(forms.contains(&"learned".to_string()));
        assert!(forms.contains(&"learning".to_string()));
    }

    #[tokio::test]
    async fn test_stem_expander_expand() {
        let expander = StemExpander::default();
        let query = create_test_query("processing algorithms");

        let expanded = expander.expand(&query).await;
        assert!(!expanded.is_empty());

        let combined_text = expanded[0].text.clone();
        assert!(combined_text.contains("processing"));
    }

    #[test]
    fn test_ngram_expander_char_ngrams() {
        let expander = NGramExpander::default();
        let ngrams = expander.char_ngrams("hello", 3);

        assert!(ngrams.contains(&"hel".to_string()));
        assert!(ngrams.contains(&"ell".to_string()));
        assert!(ngrams.contains(&"llo".to_string()));
    }

    #[test]
    fn test_ngram_expander_word_ngrams() {
        let expander = NGramExpander::default();
        let ngrams = expander.word_ngrams("machine learning algorithms", 2);

        assert!(ngrams.contains(&"machine learning".to_string()));
        assert!(ngrams.contains(&"learning algorithms".to_string()));
    }

    #[tokio::test]
    async fn test_ngram_expander_expand() {
        let expander = NGramExpander::default();
        let query = create_test_query("machine learning");

        let expanded = expander.expand(&query).await;
        assert!(!expanded.is_empty());
    }

    #[test]
    fn test_prf_stop_words() {
        let prf = PseudoRelevanceFeedback::default();

        assert!(prf.is_stop_word("the"));
        assert!(prf.is_stop_word("and"));
        assert!(!prf.is_stop_word("algorithm"));
        assert!(!prf.is_stop_word("machine"));
    }

    #[test]
    fn test_prf_extract_top_terms() {
        let prf = PseudoRelevanceFeedback::default();
        let query = create_test_query("machine learning");
        let results = create_test_results();

        let top_terms = prf.extract_top_terms(&results, &query);
        assert!(!top_terms.is_empty());

        // Should not include query words or stop words
        for (term, _) in &top_terms {
            assert_ne!(term, "machine");
            assert_ne!(term, "learning");
            assert!(!prf.is_stop_word(term));
        }
    }

    #[tokio::test]
    async fn test_prf_reformulate() {
        let prf = PseudoRelevanceFeedback::default();
        let query = create_test_query("machine learning");
        let results = create_test_results();

        let reformulated = prf.reformulate(&query, &results).await;
        assert!(reformulated.text.contains("machine learning"));
        assert!(reformulated.text.len() >= query.text.len());
    }

    #[tokio::test]
    async fn test_prf_reformulate_empty_results() {
        let prf = PseudoRelevanceFeedback::default();
        let query = create_test_query("machine learning");
        let results: Vec<SearchResult> = vec![];

        let reformulated = prf.reformulate(&query, &results).await;
        assert_eq!(reformulated.text, query.text);
    }

    #[test]
    fn test_query_reformulator_simplify() {
        let reformulator = QueryReformulator::new();
        let query = create_test_query("what is the machine learning algorithm");

        let simplified = reformulator.simplify(&query);
        assert!(!simplified.text.contains("the"));
        assert!(simplified.text.contains("machine"));
        assert!(simplified.text.contains("learning"));
    }

    #[test]
    fn test_query_reformulator_clarify() {
        let reformulator = QueryReformulator::new();
        let query = create_test_query("machine learning");

        let clarified = reformulator.clarify(&query, "neural networks");
        assert!(clarified.text.contains("machine learning"));
        assert!(clarified.text.contains("neural networks"));
    }

    #[test]
    fn test_query_reformulator_decompose() {
        let reformulator = QueryReformulator::new();
        let query = create_test_query("machine learning and deep learning");

        let decomposed = reformulator.decompose(&query);
        assert!(decomposed.len() >= 2);
    }

    #[test]
    fn test_query_reformulator_decompose_simple() {
        let reformulator = QueryReformulator::new();
        let query = create_test_query("machine learning");

        let decomposed = reformulator.decompose(&query);
        assert_eq!(decomposed.len(), 1);
        assert_eq!(decomposed[0].text, "machine learning");
    }

    #[test]
    fn test_query_reformulator_rephrase_as_question() {
        let reformulator = QueryReformulator::new();

        let query = create_test_query("machine learning");
        let rephrased = reformulator.rephrase_as_question(&query);
        assert!(rephrased.text.ends_with('?'));

        // Already a question
        let query = create_test_query("What is machine learning?");
        let rephrased = reformulator.rephrase_as_question(&query);
        assert_eq!(rephrased.text, "What is machine learning?");
    }

    #[tokio::test]
    async fn test_composite_expander() {
        let config = ExpansionConfig::default();
        let composite = CompositeExpander::default_composite(config);

        let query = create_test_query("search data");
        let expanded = composite.expand(&query).await;

        assert!(!expanded.is_empty());
    }

    #[tokio::test]
    async fn test_composite_expander_reformulate() {
        let config = ExpansionConfig::default();
        let composite = CompositeExpander::default_composite(config);

        let query = create_test_query("machine learning");
        let results = create_test_results();

        let reformulated = composite.reformulate(&query, &results).await;
        assert!(!reformulated.text.is_empty());
    }

    #[tokio::test]
    async fn test_composite_expander_deduplication() {
        let config = ExpansionConfig::default();
        let composite = CompositeExpander::default_composite(config).with_deduplication(true);

        let query = create_test_query("search");
        let expanded = composite.expand(&query).await;

        // Check for duplicates
        let mut seen: HashSet<String> = HashSet::new();
        for q in &expanded {
            let key = q.text.to_lowercase();
            assert!(!seen.contains(&key), "Duplicate found: {key}");
            seen.insert(key);
        }
    }

    #[tokio::test]
    async fn test_synonym_expander_custom_dictionary() {
        let mut dictionary = HashMap::new();
        dictionary.insert(
            "rust".to_string(),
            vec!["oxidation".to_string(), "corrosion".to_string()],
        );

        let config = ExpansionConfig::default();
        let expander = SynonymExpander::with_dictionary(config, dictionary);

        let synonyms = expander.get_synonyms("rust");
        assert!(synonyms.contains(&"oxidation".to_string()));
    }

    #[test]
    fn test_expanded_query_weight_clamping() {
        let query = create_test_query("test");
        let expanded =
            ExpandedQuery::new(query.clone(), ExpansionMethod::Synonyms).with_weight(1.5);
        assert!((expanded.weight - 1.0).abs() < f32::EPSILON);

        let expanded = ExpandedQuery::new(query, ExpansionMethod::Synonyms).with_weight(-0.5);
        assert!(expanded.weight.abs() < f32::EPSILON);
    }

    #[test]
    fn test_ngram_range_validation() {
        let config = ExpansionConfig::new().with_ngram_range(0, 2);
        assert_eq!(config.ngram_range.0, 1);

        let config = ExpansionConfig::new().with_ngram_range(3, 2);
        assert_eq!(config.ngram_range.0, 3);
        assert_eq!(config.ngram_range.1, 3);
    }

    #[tokio::test]
    async fn test_stem_expander_disabled() {
        let config = ExpansionConfig {
            enable_stemming: false,
            ..Default::default()
        };
        let expander = StemExpander::new(config);
        let query = create_test_query("processing");

        let expanded = expander.expand(&query).await;
        assert_eq!(expanded.len(), 1);
        assert_eq!(expanded[0].text, "processing");
    }

    #[test]
    fn test_char_ngrams_short_word() {
        let expander = NGramExpander::default();
        let ngrams = expander.char_ngrams("hi", 3);
        assert_eq!(ngrams, vec!["hi"]);
    }

    #[test]
    fn test_word_ngrams_short_phrase() {
        let expander = NGramExpander::default();
        let ngrams = expander.word_ngrams("hello", 2);
        assert_eq!(ngrams, vec!["hello"]);
    }
}
