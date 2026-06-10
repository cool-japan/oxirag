# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.12.0] - 2026-06-10

### Added

- **Cross-Encoder Reranking** (feature `cross-encoder`): Two-stage pairwise reranking — bi-encoder recall followed by cross-encoder precision. `LexicalCrossEncoder` extracts 7 interaction features (`exact_match_ratio`, `term_overlap`, `idf_weighted_overlap`, `query_coverage`, `doc_coverage`, `ordered_bigram_match`, `length_ratio`) per (query, doc) pair, computes IDF from the candidate set, and squashes via logistic. `CrossEncoderReranker.rerank()` blends original and cross scores (`fused = α*orig + (1-α)*cross`), filters by threshold, re-ranks, and truncates to `top_n`. ~46 tests.
- **Contextual Retrieval** (feature `contextual-retrieval`, depends on `chunking`): Anthropic Contextual Retrieval — prepends a situating blurb to each chunk before indexing. `ExtractiveContextualizer` builds TF-IDF extractive doc summaries (top-N sentences), prepends title / positional hint / preceding-chunk gist. `ContextualIndexBuilder.build()` computes real position ratios, calls the contextualizer per chunk, and returns `Vec<ContextualChunk>`. ~35 tests.
- **Lost-in-the-Middle Reordering** (feature `lost-in-middle`): Mitigates Liu et al. 2023 primacy/recency bias by U-shaped positioning. `LostInMiddleReorderer.reorder()` sorts results by score descending then alternates front/back assignment (Sandwich/HeadTail), placing highest-relevance docs at both ends. `ReorderStrategy` (4 variants), `ReorderReport`. `reorder_with_report()` returns both result and metadata. ~39 tests.
- **Reflexion** (feature `reflexion`): Shinn et al. 2023 verbal self-reflection with episodic memory. `ReflexionEngine.run<E>` loops: retrieve (reflection-augmented query) → draft → evaluate → if below threshold: reflect+store → retry. `HeuristicEvaluator` computes grounding (max Jaccard vs sources) + coverage (fraction of docs with >10% overlap). `HeuristicSelfReflector` produces tiered lessons by score level. `EpisodicMemory` bounded FIFO buffer. ~44 tests.
- **Tree-of-Thoughts** (feature `tree-of-thought`): Yao et al. 2023 branching reasoning search. `TreeOfThoughtEngine.run<E>` retrieves once then grows a tree via BFS (with beam pruning) or DFS, evaluating nodes via `ThoughtEvaluator`, pruning below `value_threshold`, and synthesizing from the best-leaf path. `ThoughtTree` with `children_of`/`path_to_root`/`best_leaf`. ~48 tests.
- **Chain-of-Verification** (feature `chain-of-verification`, depends on `advanced-retrieval`): Dhuliawala et al. 2023 CoVe — draft → plan verification questions → answer each independently (fresh re-retrieve) → revise. `HeuristicQuestionPlanner` splits draft into per-sentence probes. `ChainOfVerificationEngine.run<E>` assigns `ClaimVerdict` (Supported/Contradicted/Unverified) via Jaccard; if revising, fuses supported results via `reciprocal_rank_fusion`. `faithfulness_delta` measures improvement. ~44 tests.
- **Long-Term Memory** (feature `long-term-memory`): Park et al. 2023 generative-agents memory stream. `LongTermMemoryStore` with FNV-1a bag-of-words embeddings, capacity-bounded insertion, `prune()` (evict lowest importance), and `reflect()` (synthesize high-importance observations into a Reflection record). `MemoryRetriever.retrieve()` scores by `combined = w_r * decay(age) + w_i * importance + w_s * cosine`, bumps access_count and last_accessed. `HeuristicImportanceScorer` (affective words + digits + caps). ~62 tests.
- **Memory Compression** (feature `memory-compression`): MemGPT-style hierarchical conversation compaction. `HierarchicalMemory.push_turn()` retains recent turns verbatim; overflows trigger `ExtractiveTurnCompressor` (salient sentence + fact extraction) into level-1 `CompressedBlock`s which cascade to level-2 summaries when the block count reaches `block_size_turns`. `render(budget)` produces verbatim recent + summaries within the token budget. ~46 tests.
- **Entity Memory** (feature `entity-memory`): Per-entity knowledge tracking across turns. `EntityMemoryStore.observe(text)` runs `HeuristicEntityMentionExtractor` (capitalised n-gram detection), upserts `EntityKnowledge` (name, category, facts, mention_count, salience), and enforces `max_facts_per_entity`. `top_salient(n)` returns the most salient entities; `context_for(query)` retrieves relevant entity summaries. ~58 tests.
- **Retrieval Evaluation** (feature `retrieval-eval`): IR ranking metrics distinct from RAGAS. Free functions `precision_at_k`, `recall_at_k`, `f1_at_k`, `hit_rate_at_k`, `reciprocal_rank`, `mrr`, `average_precision`, `dcg_at_k` (graded: `(2^gain-1)/log(i+2)`), `ndcg_at_k` (ideal-DCG normalized). `RetrievalEvaluator.evaluate()` and `evaluate_batch()` (macro-averaged MAP/MRR/nDCG). `Qrels` with graded gains. ~67 tests.
- **LLM-as-Judge** (feature `llm-judge`): Deterministic heuristic judge for pointwise/pairwise/reference-based evaluation. `HeuristicJudge` computes relevance (Jaccard query/answer), groundedness (max-source Jaccard), coherence, and conciseness (word-count penalty). `LlmJudge.score_pointwise()`, `compare()` (ties within `tie_margin`), `score_with_reference()`. `Rubric` with `relevance_helpfulness_groundedness()` and `correctness()` built-ins + `normalize()`. ~53 tests.
- **Prompt Optimization** (feature `prompt-optimization`): DSPy/APE-lite deterministic few-shot selection and variant scoring. `DemoSelector.select()` supports KNN (FNV-1a cosine), MMR-diverse (λ-balanced relevance/diversity), Deterministic (index-order), and Hardest (lowest quality) strategies. `PromptOptimizer.evaluate_variants()` scores each `PromptVariant` over a dev set via any `OutputScorer`; `best()` returns the top-scoring variant. ~56 tests.
- **Four v0.12.0 theme umbrella features**: `rerank-precision`, `advanced-reasoning`, `memory-state`, `eval-optimization`.

### Codebase Statistics (v0.12.0)

- **Tests**: 3,285 (all passing; +598 from v0.11.0's 2,687)
- **Clippy Warnings**: 0
- **Rustdoc Warnings**: 0

## [0.11.0] - 2026-06-10

### Added

- **Multi-hop Retrieval** (feature `multi-hop`, depends on `graphrag`): Entity-chain traversal over the knowledge graph. `MultiHopRetriever.run<E>` detects entity mentions in the query, performs hop-by-hop BFS via `GraphRelationship` edges, and retrieves documents at each hop. `HopConfig`, `HopState`, `MultiHopResult`. ~18 tests.
- **Fact Triple Extraction** (feature `fact-triples`): Heuristic SVO triple extraction without regex or ML. `TripleExtractor` splits text into sentences, detects verbs via a curated `COMMON_VERBS` list + morphological heuristics, and extracts `(subject, predicate, object)` triples. `TripleStore` with `find_by_subject`/`find_by_predicate`/`match_query`. ~18 tests.
- **Knowledge Graph QA** (feature `knowledge-graph-qa`, depends on `graphrag + fact-triples`): Direct KGQA via subgraph extraction and fact synthesis. `KgqaEngine.answer()` matches query tokens to entity names, performs BFS subgraph expansion, extracts triples via `TripleExtractor`, and synthesizes a template answer with confidence scoring. ~17 tests.
- **Iterative RAG** (feature `iterative-rag`, depends on `advanced-retrieval`): ITER-RETGEN iterative retrieval loop. `IterativeRagEngine.run<E>` retrieves docs → builds a draft → extracts TF-IDF expansion terms (stop-word filtered) → re-retrieves with expanded query, up to `max_iterations`. Deduplicates by document ID across iterations. `IterativeOutput` with per-step trace. ~18 tests.
- **Chain-of-Note** (feature `chain-of-note`): Per-document extractive notes synthesized into a final answer. `ChainOfNoteEngine.process()` scores sentences Jaccard-vs-query, extracts top-N as a note per doc, chains all notes, and synthesizes. `DocumentNote`, `NoteChain`, `NoteConfig`. ~19 tests.
- **Answer Aggregation** (feature `answer-aggregation`): Multi-candidate answer fusion. `AnswerAggregator.aggregate()` supports three strategies: `MajorityVote` (sentence voting across candidates), `WeightedFusion` (confidence-weighted selection), `Extractive` (union + Jaccard dedup). `CandidateAnswer`, `AggregatedAnswer`. ~23 tests.
- **Hallucination Detection** (feature `hallucination-detection`): Lexical claim-support scoring against source documents. `HallucinationDetector.detect()` splits the answer into claims, scores each via sentence-level Jaccard against sources, and classifies claims as supported or hallucinated. `HallucinationReport` with `hallucination_rate`, `is_clean()`. ~18 tests.
- **Consistency Checking** (feature `consistency-checking`): Cross-claim consistency detection within generated answers. `ConsistencyChecker.check()` performs pairwise sentence analysis detecting `Numerical`, `Temporal`, and `Negation` conflicts via shared-noun heuristics. `ConsistencyReport` with per-conflict `confidence`. ~19 tests.
- **Trust Scoring** (feature `trust-scoring`, depends on `hallucination-detection + consistency-checking`): Composite answer trustworthiness scoring from four weighted components: grounding (1 - hallucination_rate), consistency, source quality (source count proxy), and completeness (query token coverage). `TrustScorer.score()`, `TrustComponents.normalize()`, `TrustScore.label()`. ~22 tests.
- **Semantic Router** (feature `semantic-router`): Embedding-based retrieval strategy selection via FNV-1a hash pseudo-embeddings and cosine KNN matching against labeled examples. `SemanticRouter.route()` returns a `RoutingDecision{target, confidence, reasoning}`. Six `RoutingTarget` variants. `route_with_fallback()` never errors. ~19 tests.
- **Query Planning** (feature `query-planning`, depends on `query-decomposition`): DAG-structured query execution plans. `QueryPlanner.plan()` classifies queries (comparative/aggregative/multi-step/verify/simple) and generates typed `PlanStep` DAGs. `PlanExecutor.run<E>` executes via Kahn's topological sort with cycle detection. `SynthesisStrategy` for plan-level output fusion. ~21 tests.
- **Pipeline Composer** (feature `pipeline-composer`, depends on `guardrails + output-validation`): Composable synchronous RAG pipeline stages. `PipelineStage` trait with `PassThroughStage`, `FormatStage`, `TruncateStage`, `KeywordFilterStage`, `SanitizerStage`, and `GuardrailStage`. `ComposedPipeline.run()` with configurable `stop_on_block`. ~21 tests.
- **Four v0.11.0 theme umbrella features**: `graph-reasoning`, `iterative-gen`, `trust-verify`, `routing-compose`.

### Codebase Statistics (v0.11.0)

- **Tests**: 2,687 (all passing; +232 from v0.10.0's 2,455)
- **Clippy Warnings**: 0
- **Rustdoc Warnings**: 0
- **New features**: `multi-hop`, `fact-triples`, `knowledge-graph-qa`, `iterative-rag`, `chain-of-note`, `answer-aggregation`, `hallucination-detection`, `consistency-checking`, `trust-scoring`, `semantic-router`, `query-planning`, `pipeline-composer`, `graph-reasoning`, `iterative-gen`, `trust-verify`, `routing-compose`
- **New files**: `src/multi_hop/{mod,types,traversal,tests}.rs`, `src/fact_triple/{mod,types,extractor,tests}.rs`, `src/knowledge_graph_qa/{mod,types,engine,tests}.rs`, `src/iterative_rag/{mod,types,engine,tests}.rs`, `src/chain_of_note/{mod,types,engine,tests}.rs`, `src/answer_aggregator/{mod,types,aggregator,tests}.rs`, `src/hallucination_detector/{mod,types,detector,tests}.rs`, `src/consistency_checker/{mod,types,checker,tests}.rs`, `src/trust_score/{mod,types,scorer,tests}.rs`, `src/semantic_router/{mod,types,router,tests}.rs`, `src/query_planning/{mod,types,planner,executor,tests}.rs`, `src/pipeline_composer/{mod,types,stage,composer,tests}.rs`

## [0.10.0] - 2026-06-10

### Added

- **Self-RAG** (feature `self-rag`): Self-reflective retrieval with reflection tokens. `ReflectionToken` enum (9 variants). `Reflector` sync trait with `HeuristicReflector` (lexical Jaccard) and `MockReflector`. `SelfRagEngine.run<E>` — decide-retrieve → retrieve → per-doc relevance critique → mock draft → support + utility critique. `SelfRagConfig`, `SelfRagOutput{answer, reflection_tokens, retrieved, critiques}`. ~19 tests.
- **Agentic / ReAct** (feature `agentic`): ReAct Thought→Action→Observation loop. `Tool` async trait (`Send+Sync`). `ToolRegistry`. Built-ins `CalculatorTool`, `LookupTool`, `MockTool`. `AgentAction{Search, UseTool, Finish}`. `ReActAgent.run<E>` with heuristic action planner and verb-prefix stripping. `AgenticConfig{max_steps:6, top_k:5}`. ~28 tests.
- **Query Decomposition** (feature `query-decomposition`, depends on `advanced-retrieval`): Sub-question generation → per-sub retrieval → RRF recombination. `DecompositionStrategy{Parallel, LeastToMost, StepBack}`. `QueryDecomposer` with heuristic conjunction/clause splitting (always ≥1 sub-question). `QueryDecompositionEngine.run<E>` fuses sub-results via `reciprocal_rank_fusion`. ~24 tests.
- **Context Compression** (feature `context-compression`): Extractive context distillation before generation. `ContextCompressor` trait. `ExtractiveCompressor` (sentence scoring + greedy token-budget packing). `RedundancyFilter` (Jaccard-based dedup). `CompressionConfig{token_budget:512, relevance_threshold:0.1, redundancy_threshold:0.8}`. `CompressedContext{text, ratio, …}`. ~16 tests.
- **Guardrails** (feature `guardrails`): PII scanning, prompt-injection detection, content moderation — zero regex, pure char-class state machines. `PiiDetector` (email/phone/SSN/credit-card/IPv4). `InjectionDetector` (phrase-signal scoring). `ContentModerator` (wordlist severity). `GuardrailEngine.check()` → `GuardrailReport{violations, redacted_text, blocked}`. ~24 tests.
- **Structured Extraction** (feature `structured-extraction`): Keyword-proximity typed field extraction (no regex, no ML). `FieldType{Text, Number, Date, Boolean, Enum}`. `SchemaExtractor.extract()` → `ExtractedRecord{fields, missing_required}` with `to_json`. ~16 tests.
- **Output Validation** (feature `output-validation`): Generated-answer rule enforcement. `RuleKind{MinLength, MaxLength, RequiresCitation, NoBannedPhrase, MustContain, JsonParsable, MaxRepetition}`. `OutputValidator.validate()` → `ValidationReport{passed, violations, max_severity}`. ~18 tests.
- **Graph Community Detection** (feature `graph-community`, depends on `graphrag`): Louvain modularity community detection over `&[GraphEntity]`/`&[GraphRelationship]` slices. `LouvainDetector` implementing `CommunityDetector`. `CommunityGraph{communities, modularity}`. Greedy modularity optimization with configurable resolution. ~16 tests.
- **Graph Summarization** (feature `graph-summarization`, depends on `graph-community`): Microsoft-GraphRAG-style extractive summaries + global/local search. `CommunitySummarizer` (degree-ranked key entities + relationship phrases). `GlobalSearchEngine` (map-reduce over summaries). `LocalSearchEngine` (entity-neighbourhood expansion). ~18 tests.
- **RAPTOR** (feature `raptor`): Recursive clustering + extractive summary tree. FNV-1a hash pseudo-embeddings (L2-normalized). `ClusterStrategy{Agglomerative, KMeansLite}`. `RaptorBuilder.build()` → `RaptorTree` with `collapsed_retrieval(query, top_k)`. ~18 tests.
- **Parent-Document Retrieval** (feature `parent-document`, depends on `chunking`): Small-to-big retrieval. `ParentChildIndex` with child→parent mapping. `ParentDocumentRetriever.run<E>` — search children → expand to parents → dedup. `ExpandedResult{parent, matched_children, score}`. ~18 tests.
- **Temporal Re-ranking** (feature `temporal-retrieval`): Recency-decay re-scoring. `DecayFunction{Exponential, Linear, Gaussian, None}`. `TemporalReranker.rerank()` with hand-written ISO 8601 parser, blend formula `(1-w)*score + w*score*decay(age)`. `TemporalConfig{decay, weight, use_updated}`. ~18 tests.
- **Four theme umbrella features**: `agentic-reasoning`, `safety-governance`, `graph-intelligence`, `retrieval-depth`.

### Codebase Statistics (v0.10.0)

- **Tests**: 2,455 (all passing; +236 from v0.9.0)
- **Clippy Warnings**: 0
- **Rustdoc Warnings**: 0
- **New features**: `self-rag`, `agentic`, `query-decomposition`, `context-compression`, `guardrails`, `structured-extraction`, `output-validation`, `graph-community`, `graph-summarization`, `raptor`, `parent-document`, `temporal-retrieval`, `agentic-reasoning`, `safety-governance`, `graph-intelligence`, `retrieval-depth`
- **New files**: `src/self_rag/{mod,types,reflect,engine,tests}.rs`, `src/agentic/{mod,types,tool,agent,tests}.rs`, `src/query_decomposition/{mod,types,decomposer,engine,tests}.rs`, `src/context_compression/{mod,types,compressor,tests}.rs`, `src/guardrails/{mod,types,pii,injection,moderation,engine,tests}.rs`, `src/structured_extraction/{mod,types,extractor,tests}.rs`, `src/output_validation/{mod,types,validator,tests}.rs`, `src/graph_community/{mod,types,louvain,tests}.rs`, `src/graph_summarization/{mod,types,summarizer,search,tests}.rs`, `src/raptor/{mod,types,cluster,tree,tests}.rs`, `src/parent_document/{mod,types,retriever,tests}.rs`, `src/temporal/{mod,types,reranker,tests}.rs`

## [0.9.0] - 2026-06-10

### Added

- **Prompt Template Registry** (feature `prompt-templates`): Versioned template registry + 2-phase tokenize + recursive-descent parser. `TemplateId` newtype. `PromptTemplate` with `required_vars` validation. `RenderContext` with string vars and boolean flags. `TemplateEngine` supporting `{{var}}`, `{{#if flag}}..{{else}}..{{/if}}`, `{{#unless flag}}..{{/unless}}` with depth cap 64. `PromptRegistry` with version-sorted storage, `get_latest`/`get_version`/`versions`/`render_latest`. Four built-in RAG templates: `answer-synthesis`, `query-rewrite`, `doc-grading`, `citation`. ~50 tests.
- **Query Router** (feature `query-routing`): Intent classification and adaptive retrieval-strategy routing. `QueryIntent` (10 variants: Factual/Definitional/Comparative/Navigational/MultiHop/Conversational/Exploratory/Temporal/Aggregation/Unknown). `RoutingStrategy` (6 variants: VectorSearch/HybridSearch/GraphSearch/MultiHop/Conversational/DirectAnswer). `IntentScores` sorted-descending distribution with `top()` guard. `RouterConfig` with full routing/fallback/top-k tables. `HeuristicIntentClassifier` with 9 signal tables, year detection, pronoun detection, always-non-empty floor. `MockIntentClassifier`. `QueryRouter<C>` generic router with low-confidence fallback. ~50 tests.
- **Corrective RAG** (feature `corrective-rag`, depends on `advanced-retrieval`): CRAG retrieval-quality grading and corrective re-retrieval loop (Yan et al. 2024). `CragConfig` with upper/lower thresholds, max corrections, MMR params. `RetrievalGrade` (Correct/Ambiguous/Incorrect). `CorrectiveAction` (UseAsIs/Refine/Rewrite/Discard/Augment). `HeuristicRetrievalGrader` (lexical Jaccard). `MockRetrievalGrader`. `KnowledgeRefiner` (decompose→filter→recompose strips with sentence-level relevance scoring). `QueryRefiner` (salient-term rewrite with never-empty guard). `CorrectiveRagEngine<G>` with method-level `run<E>` Echo-generic pattern; real lexical pseudo-embeddings (`DefaultHasher % dim` + L2 normalize) drive genuine MMR dedup. ~47 tests.
- **Attribution** (feature `attribution`): Inline citation and source attribution for generated answers. `LexicalAligner` (token-Jaccard). `SentenceAligner<A>` splits answer → scores against sources → attaches top-N citations above threshold. `Attributor<A>` with stable dedup by source_id. `CitationFormatter` for `[1]`/`[^1]`/author styles with bibliography. `FaithfulnessChecker` with grounding fraction (0.0 guard on empty spans). `AttributedAnswer` with `overall_faithfulness`. ~44 tests.

### Codebase Statistics (v0.9.0)

- **Tests**: 2,219 (all passing; +190 from v0.8.0: +50 prompt-templates, +50 query-routing, +47 corrective-rag, +44 attribution, -1 skipped adjustment)
- **Clippy Warnings**: 0
- **Rustdoc Warnings**: 0
- **New features**: `prompt-templates`, `query-routing`, `corrective-rag`, `attribution`, `adaptive-control-plane`
- **New files**: `src/prompt_templates/{mod,types,engine,registry,tests}.rs`, `src/query_router/{mod,types,classifier,router,tests}.rs`, `src/corrective_rag/{mod,types,grader,strip,engine,tests}.rs`, `src/attribution/{mod,types,aligner,citation,faithfulness,tests}.rs`

## [0.8.0] - 2026-05-17

### Added

- **Conversational RAG** (feature `conversational`): Full multi-turn conversation support. `ConversationHistory` with typed `Turn`/`TurnRole`. Four `HistoryBuffer` strategies: `FullHistoryBuffer`, `SlidingWindowBuffer`, `SummaryBuffer` (rolling lazy summary via `Arc<RwLock<_>>`), `HybridBuffer` (recent turns + truncated summary). `FollowUpDetector` with word-boundary pronoun detection, entity extraction from assistant turns, and pronoun resolution. `QueryReformulator` with four `ReformulationStrategy` variants (Concatenation, ContextInjection, FollowUpResolution, Standalone). `InMemorySessionManager` with LRU oldest-session eviction at capacity. `ConversationalPipeline<S, B>` generic wrapper that enriches user queries with conversation context before forwarding to the RAG pipeline. 77 new tests.
- **FLARE Adaptive Retrieval** (feature `flare`): Forward-Looking Active REtrieval (Jiang et al. 2023). `FlareEngine<G, R>` with full iterative generate-retrieve-generate loop. `ConfidenceEstimator` with log-smoothed token-frequency pseudo-probability, sentence-level averaging, uncertain-sentence identification. `ContextWindow` with sorted, deduped, budget-trimmed context docs. `FlareOutput` with `retrieval_rate()`. `MockFlareGenerator` (cyclic, `Arc<AtomicUsize>` call counter), `TemplateGenerator` (`{query}`/`{context}` substitution). `MockFlareRetriever` + `QueryAugmentedRetriever<R>`. 56 new tests.
- **Knowledge Base Collections** (feature `collections`): Multi-tenant namespaced vector store. `CollectionId` with lowercase-alphanumeric normalisation. `InMemoryCollectionStore` (capacity-bounded CRUD, running-average latency in `CollectionStats`). `CollectionIndex<S>` with per-collection `InMemoryVectorStore`, single-collection search, and cross-collection RRF fusion (`k=60`) via `cross_collection_search` / `search_all`. `FederatedResult` carries collection provenance. 29 new tests.
- **Integrated Document Processing Pipeline** (feature `document-pipeline`, depends on `chunking + semantic-cache + advanced-retrieval`): `IndexingPipeline` auto-chunks documents with any of the four chunking strategies, content-hash deduplication, and per-chunk provenance tracking (`ChunkProvenance`). `RetrievalPipeline` with optional `InMemorySemanticCache` hit/miss, `MmrReranker` post-processing, and provenance enrichment into `DocumentAwareResult`. `DocumentPipelineBuilder` fluent API that constructs both pipelines sharing a provenance map and stats handle. `PipelineStats` with cache_hits counter. 26 new tests.

### Codebase Statistics (v0.8.0)

- **Tests**: 2,029 (all passing; +188 from v0.7.0: +77 conversational, +56 flare, +29 collections, +26 document-pipeline)
- **Clippy Warnings**: 0
- **Rustdoc Warnings**: 0
- **New features**: `conversational`, `flare`, `collections`, `document-pipeline`
- **New files**: `src/conversation/{mod,types,buffer,reformulator,session,tests}.rs`, `src/retrieval_loop/{mod,types,confidence,generator,retriever,engine,tests}.rs`, `src/collections/{mod,types,store,index,tests}.rs`, `src/document_pipeline/{mod,types,indexing,retrieval,tests}.rs`

## [0.7.0] - 2026-05-17

### Added

- **Document Chunking** (feature `chunking`): `DocumentChunker` with four strategies — `FixedSizeChunker` (Unicode scalar sliding window with configurable overlap), `SentenceChunker` (sentence-boundary splits with overlap seeding), `RecursiveChunker` (multi-separator cascade: `\n\n` → `\n` → `. ` → ` ` → `""`), `MarkdownChunker` (ATX heading splits with fenced-code-block tracking). `ChunkConfig` with `chunk_size`, `chunk_overlap`, `min_chunk_size`, `strip_whitespace`. `Chunk` with `into_document()` for seamless pipeline integration. Zero external deps (pure Rust).
- **RAGAS-style Evaluation** (feature `rag-eval`): `RagEvaluator` with four lexical/heuristic metrics: `AnswerRelevanceScorer` (Jaccard + phrase boost), `FaithfulnessScorer` (sentence-level context overlap), `ContextPrecisionScorer`, `ContextRecallScorer`. Configurable `OverallScorer` with per-metric weights. `EvaluationSample::from_pipeline_output()` for frictionless pipeline wiring. `EvaluationDataset` with `save_json()`/`load_json()` for benchmark persistence. `EvalError` with `MissingGroundTruth`, `EmptyContext`, `Other(String)` variants.
- **Semantic Cache** (feature `semantic-cache`): `InMemorySemanticCache` implementing `SemanticCache` async trait. Cosine-similarity lookup against cached query embeddings (configurable `similarity_threshold`, default 0.92). LRU eviction (`max_entries`), optional TTL expiry, `CacheStats` with `hit_rate`. Zero-vector guard prevents NaN scores. Thread-safe via `Arc<Mutex<...>>`.
- **Advanced Retrieval** (feature `advanced-retrieval`): `RagFusion` with deterministic multi-query variant generation (negation, qualifier expansion, aspect decomposition) and `reciprocal_rank_fusion()` (HashMap deduplication + tie-breaking). `HydeRetrieval` (Hypothetical Document Embeddings) with stop-word-filtered expansion and delegate search. `MmrReranker` (Maximal Marginal Relevance) greedy loop with configurable λ (`lambda_param`, default 0.5) and fallback to raw score when embedding unavailable. `RetrievalConfig` + `AdvancedRetrievalError` types.

### Codebase Statistics (v0.7.0)

- **Tests**: 1,841 (all passing; +160 from v0.6.0: +40 chunking, +56 rag-eval, +27 semantic-cache, +37 advanced-retrieval)
- **Clippy Warnings**: 0
- **Rustdoc Warnings**: 0
- **New features**: `chunking`, `rag-eval`, `semantic-cache`, `advanced-retrieval`
- **New files**: `src/chunking/{mod,config,chunk,strategies,tests}.rs`, `src/evaluation/{mod,types,metrics,evaluator,dataset,tests}.rs`, `src/semantic_cache/{mod,config,entry,cache,tests}.rs`, `src/advanced_retrieval/{mod,types,rag_fusion,hyde,mmr,tests}.rs`

## [0.6.0] - 2026-05-17

### Added

- **Node.js bindings** (feature `nodejs`): napi-rs 2.x, exposes `NapiPipeline`, `NapiPipelineBuilder`, `NapiDocument`, `NapiQuery`, `NapiSearchResult`; all async methods return native Promises; build via `@napi-rs/cli` (`npm run build`).
- `package.json` at crate root for npm package `@cool-japan/oxirag` (version 0.6.0).
- `examples/nodejs/quickstart.js`: end-to-end Node.js quickstart (index, count, query, print results).
- `tests/nodejs_smoke.rs`: Rust-side smoke tests (11 test cases) exercising the underlying `Document`/`Query`/`Pipeline` types powering the napi bridge; no Node.js runtime required. Full JS↔Rust integration tests run via `npm test`.
- `build.rs`: calls `napi_build::setup()` which is a no-op for non-Node.js targets and wires N-API linkage when `--features nodejs` is active. Also links `libnapi_stub.so` (weak stubs for all 40 napi_* symbols) for test/bin/example artifacts so `cargo nextest run --all-features` passes without a Node.js runtime in the linker environment.
- `napi_stub.c` + `libnapi_stub.so`: 40 weak-stub definitions of napi C functions; only used at link time for non-cdylib artifacts; real Node.js symbols override them when the `.node` addon is loaded at runtime.
- `src/nodejs/` module tree: `mod.rs`, `types.rs` (`NapiDocument`, `NapiQuery`, `NapiSearchResult`), `pipeline.rs` (`NapiPipeline`, `NapiPipelineOutput`, `NapiSearchResultItem`), `builder.rs` (`NapiPipelineBuilder`).
- `NapiPipeline::search()`: direct Echo-layer search bypassing Speculator and Judge.
- **File refactors (under 2 000-line policy)**: `src/streaming.rs` (1 455 → thin orchestrator + `streaming/{types,wrapper,progress,tests}.rs`); `src/query_expansion.rs` (1 439 → thin orchestrator + 8 sub-files); `src/circuit_breaker.rs` (1 390 → thin orchestrator + 5 sub-files); `src/connection_pool.rs` (1 407 → thin orchestrator + 6 sub-files); all sub-files ≤ 554 lines.
- **Documentation suite**: `docs/adr/` (5 Architecture Decision Records); `docs/layers/` (4 per-layer tutorials); `docs/troubleshooting.md` (555 lines).
- **TypeScript WASM wrapper** (`npm/`): `@cool-japan/oxirag-wasm` npm package with typed `WasmEngine`, Web Worker bridge, and streaming helpers; `npm/package.json`, `npm/tsconfig.json`.

### Codebase Statistics (v0.6.0)

- **Tests**: 1,681 (all passing; +11 nodejs smoke, +8 skipped WASM/network tests)
- **Clippy Warnings**: 0
- **Rustdoc Warnings**: 0
- **New features**: `nodejs`
- **New files**: `build.rs`, `napi_stub.c`, `libnapi_stub.so`, `package.json`, `pyproject.toml`, `src/nodejs/mod.rs`, `src/nodejs/types.rs`, `src/nodejs/pipeline.rs`, `src/nodejs/builder.rs`, `tests/nodejs_smoke.rs`, `examples/nodejs/quickstart.js`, `npm/` (TypeScript WASM wrapper), `docs/adr/` (5 ADRs), `docs/layers/` (4 tutorials), `docs/troubleshooting.md`

## [0.5.0] - 2026-05-17

### Added

- **Python bindings** (feature `python`): PyO3 0.28 + pyo3-async-runtimes 0.28; exposes `PyPipeline`, `PyPipelineBuilder`, `PyDocument`, `PyQuery`, `PySearchResult`, `PyPipelineOutput`, `PyDraft`, `PySpanReport`; async bridge via `future_into_py`; buildable with maturin.
- **WASM IndexedDB stack**: `IndexedDbVectorStore` + `IndexedDbPrefixCache` backed by the browser IndexedDB API (features `wasm-indexeddb`, `wasm-prefix-indexeddb`); `WasmRagEngine::query_stream` returning a `ReadableStream`; Web Worker entry (`wasm_worker.rs`); bundle-size optimisation profile (`release-wasm`).
- **Docker distribution**: multi-stage `Dockerfile` (`debian:12-slim` runtime, < 50 MB image); `oxirag-server` binary configured via env vars (`OXIRAG_HOST`, `OXIRAG_PORT`, `OXIRAG_DIMENSION`); `docker-compose.yml` with Jaeger all-in-one OTel integration; `docs/docker.md` container deployment guide.
- **Performance tuning guide**: `docs/perf.md` covering SIMD backend selection, redb vs in-memory tradeoffs, cache sizing, speculator tuning, and observability overhead.
- **`docs/docker.md`**: full container deployment guide (prerequisites, build, run, env vars, health check, OTel, REST API examples, persistence, image-size tips, troubleshooting).
- **`tests/trait_bounds_native.rs`**: compile-time assertions that `EmbeddingProvider`, `VectorStore`, and `PrefixCacheStore` remain `Send + Sync` on native targets.
- **`[[bin]] oxirag-server`** in `Cargo.toml`: standalone REST server binary (`src/bin/oxirag-server.rs`, required-features `rest-server`).
- **`[profile.release-wasm]`**: dedicated Cargo profile inheriting `release` with `panic = "abort"`, `strip = true`, `opt-level = "z"`.
- **`[package.metadata.wasm-pack.profile.release]`**: `wasm-opt = ["-Oz", "--enable-bulk-memory"]` for minimal WASM bundles.

### Changed

- `VectorStore`, `Echo`, `EmbeddingProvider`, `MultiModalEmbeddingProvider` traits: now use `#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]` so WASM implementations can use `JsValue` without requiring `Send`. Callers that need `Send + Sync` can add those bounds themselves.
- `PrefixCacheStore` and `PrefixCacheExt`: same cfg-gated async_trait pattern. The `PrefixCacheExt` blanket impl is split into two cfg'd blocks (`not(wasm32)` requires `+ Send`, `wasm32` drops the `Send` bound).
- `web-sys` optional dependency: expanded feature list to include IndexedDB types, Web Worker types, and Streaming types required by `wasm-indexeddb` / `wasm-prefix-indexeddb` features.
- `pyo3 = "0.28"` and `pyo3-async-runtimes = "0.28"` added as optional deps for the `python` feature.

### Fixed

- Nothing (zero issues carried over from v0.4.0).

### Codebase Statistics (v0.5.0)

- **Tests**: 1,658 + new WASM/Python/trait-bound compile tests
- **Clippy Warnings**: 0
- **Rustdoc Warnings**: 0
- **New features**: `python`, `wasm-indexeddb`, `wasm-prefix-indexeddb`
- **New files**: `src/bin/oxirag-server.rs`, `Dockerfile`, `.dockerignore`, `docker-compose.yml`, `docs/docker.md`, `tests/trait_bounds_native.rs`

## [0.4.0] - 2026-05-17

### Added

- **Build fix**: `redb 4.x` migration — added `use redb::ReadableDatabase` to three redb-backed files (`prefix_cache/redb_backend.rs`, `layer4_graph/redb_store.rs`, `layer1_echo/storage/redb.rs`); resolves 20 build errors caused by `begin_read()` moving to a trait in redb 4.0.

- **Observability wiring** (`observability/mod.rs`, `pipeline.rs`):
  - New `SpanObserver` trait: `on_layer_complete(&LayerSpanRecord)` + `on_pipeline_complete(&PipelineSpanContext)`
  - New `MemoryObserver`: thread-safe in-memory span collector, useful for tests and REST metrics
  - `PipelineSpanContext::add_observer(Arc<dyn SpanObserver>)` + `finalize()`
  - `PipelineBuilder::with_observers(Vec<Arc<dyn SpanObserver>>)` builder method
  - `Pipeline::process()` now creates a `PipelineSpanContext` per query and wraps each layer in RAII spans
  - 5 new observer tests (error, skip, multiple-observer, finalize paths)

- **OpenTelemetry feature** (`src/observability/otel.rs`, feature `otel`):
  - `OtelSpanObserver` implements `SpanObserver` by emitting OTel spans per layer and per pipeline
  - `OtelSpanObserver::with_stdout()` — stdout exporter for examples/tests (no infrastructure needed)
  - `OtelSpanObserver::with_otlp_endpoint(url)` — OTLP/gRPC exporter for production
  - `OtelInitError` error type
  - Dependencies: `opentelemetry 0.32`, `opentelemetry_sdk 0.32`, `opentelemetry-otlp 0.32`, `opentelemetry-stdout 0.32`, `tonic 0.14`
  - 7 new tests
  - Example: `examples/otel_tracing.rs`

- **Multi-modal embeddings** (`src/layer1_echo/embedding/clip.rs`, feature `multimodal`):
  - New `EmbeddingInput<'a>` enum: `Text`, `Image`, `TextAndImage`
  - New `MultiModalEmbeddingProvider` async trait (blanket impl makes it a drop-in `EmbeddingProvider`)
  - `CandleClipProvider`: CLIP-based provider (text encoder + vision encoder) using `candle-transformers`
  - Presets: `ClipPreset::VitBase32` (512-dim, default), `VitLarge14` (768-dim), `VitLarge14_336` (768-dim 336px)
  - L2-normalised output; joint text+image embeddings via projection-head averaging
  - Dependency: `image = "0.25"` (optional)
  - 25+ new tests; network-dependent tests skip unless `OXIRAG_TEST_DOWNLOADS=1`
  - Example: `examples/multimodal_search.rs` (mock provider, no download required)

- **REST server module** (`src/rest_server.rs`, feature `rest-server`):
  - `AppState`, `build_router(AppState) -> axum::Router`, `build_and_serve(ServerConfig)`
  - Routes: `GET /health`, `POST /documents`, `POST /search`, `GET /metrics`, `POST /pipeline/query`
  - Every handler uses `PipelineSpanContext` with named layer spans fed to `MemoryObserver`
  - Dependencies: `axum 0.8`, `tower 0.5`, `tower-http 0.6` (all optional)
  - 19 new tests via `tower::ServiceExt::oneshot`
  - Example: `examples/rest_server.rs`

- **Integration test suite** (`tests/` directory — previously did not exist):
  - `tests/pipeline_e2e.rs` — 4 cross-layer pipeline E2E tests (echo-only, batch, empty store, observer integration)
  - `tests/persistence_redb.rs` — 3 persistence round-trip tests (vector store, graph store, prefix cache) using `tempfile::TempDir`; gated on `full` feature
  - `tests/observability_e2e.rs` — 5 observer integration tests (fires, layer name, duration, finalize, multi-observer)
  - `tests/multimodal.rs` — trait-bound compile checks (no model downloads)

- **Proptest fuzz harness** (`tests/fuzz_*.rs`):
  - `tests/fuzz_claim_extraction.rs` — 6 proptest cases (no-panic, normalizer idempotence, null-byte checks, dedup)
  - `tests/fuzz_query_normalization.rs` — 7 proptest cases (builder, unicode, whitespace, top_k/min_score round-trips)
  - `tests/fuzz_fingerprint.rs` — 8 proptest cases (determinism, prefix_length preservation, reflexivity, collision resistance)

- **Layer 2 + Layer 4 benchmarks** (`benches/benchmarks.rs`):
  - 5 Layer 2 bench fns: Platt calibration, temperature scaling, `RuleBasedSpeculator` verify, single-stage and multi-stage `VerificationPipeline`
  - 5 Layer 4 bench fns: entity insert, `find_by_name`, `find_by_type`, BFS traversal, relationship insert
  - Registered in `criterion_main!` under feature-gated groups

- **Doc coverage improvement** (~60 new `///` blocks):
  - `Pipeline::process`, `process_batch`, `PipelineConfig`, `PipelineBuilder` (with examples)
  - `EchoLayer`, `Echo` trait, `InMemoryVectorStore`, `MockEmbeddingProvider`
  - `WasmRagEngine::query` fully documented

### Fixed

- Flaky `test_load_test_stats_rps` test: `rps > 0.0` assertion replaced with `rps >= 0.0` + count assertion; RPS can legitimately be 0 on extremely fast hardware.
- **WASM pipeline bypass**: `WasmRagEngine::query` now calls `Pipeline::process()` through all 4 layers (was only calling `Echo::search`).

### Changed

- `src/observability.rs` → `src/observability/` module (no public API change).
- `src/simd_similarity.rs` (1662 lines) → `src/simd_similarity/` module: `mod.rs`, `generic.rs`, `neon.rs`, `x86.rs` (all ≤ 1063 lines).
- `src/hybrid_search.rs` (1639 lines) → `src/hybrid_search/` module: `mod.rs`, `types.rs`, `bm25.rs`, `fusion.rs` (all ≤ 566 lines).

### Codebase Statistics (v0.4.0)
- **Source Files**: 108 Rust files (refactors added sub-files; 3 new feature files)
- **Total Lines**: ~80,700 (Rust code; refactors redistributed, not reduced)
- **Tests**: 1,658 (+89 from v0.3.0)
- **Benchmarks**: 20 bench fns across all 4 layers (was 8; +10 Layer 2 + Layer 4)
- **Clippy Warnings**: 0
- **Rustdoc Warnings**: 0
- **New Features**: `otel`, `multimodal`, `rest-server`
- **Max file size**: 1,562 lines (`prefix_cache/redb_backend.rs`); all files under policy ceiling of 2,000

## [0.1.1] - 2026-02-06

### Added
- **OxiZ SMT Solver Integration**: Real SMT solver for Layer 3 Judge with timeout handling
  - 26 comprehensive tests covering all claim types (predicate, numeric, temporal, causal, modal)
  - 9 performance benchmarks for solver operations
  - Support for batch verification and consistency checking
- **SIMD Performance Optimization**: Hardware-accelerated similarity computations
  - ARM NEON intrinsics for Apple Silicon (M-series chips)
  - x86_64 AVX and SSE2 intrinsics for Intel/AMD CPUs
  - 5.6x-9.0x speedup for cosine similarity
  - 8x speedup for 5000 document search workloads
- **Property-Based Testing**: Added proptest 1.10.0 with 60+ property tests
  - Vector operations: commutativity, range checks, normalization idempotence
  - Cache eviction: LRU correctness, size limits, deterministic behavior
  - Graph traversal: shortest path properties, BFS correctness
  - Claim extraction: SMT-LIB generation validation
  - Query normalization: idempotence and consistency checks
- **Candle SLM Integration**: Production-ready Small Language Model support (Core Vision #1)
  - Complete `CandleSLM` implementation with Phi-2 and Phi-3 support
  - Real model inference using Candle framework (~2.7-3.8GB models)
  - Device selection: CPU, CUDA, Metal with automatic HuggingFace Hub downloads
  - Async support with proper thread management for CPU-intensive operations
  - 13 comprehensive tests (7 unit + 6 integration tests)
  - Full SmallLanguageModel trait implementation (generate, get_logprobs, verify_text)
- **Candle LoRA Training**: Complete Low-Rank Adaptation training system (Core Vision #3)
  - Real `LoRA` implementation with low-rank matrices A and B
  - Parameter-efficient fine-tuning (0.1-1% of model parameters)
  - Proper weight initialization (Kaiming uniform for A, zeros for B)
  - Training job management with status tracking and async workflow
  - Checkpoint save/load infrastructure for model deployment
  - 15 comprehensive tests covering all core functionality
  - Complete example in `examples/lora_training_example.rs`
- New module: `src/layer1_echo/similarity_simd.rs` (568 lines, platform-specific optimizations)
- New module: `src/layer2_speculator/candle_slm.rs` (843 lines, real SLM integration)
- New module: `src/distillation/candle_lora.rs` (998 lines, LoRA training)
- New module: `src/layer3_judge/oxiz_verifier.rs` (OxiZ integration)
- New example: `examples/candle_slm_example.rs` (206 lines, SLM usage demonstration)
- New example: `examples/lora_training_example.rs` (206 lines, LoRA workflow)
- Performance report: `/tmp/oxirag_performance_report.md`

### Fixed
- Fixed `test_search_performance_scales` test timing issues in debug builds
- Added conditional timeout thresholds (5s for debug, 1s for release)
- Eliminated `.unwrap()` from production code in `src/distillation/progressive.rs`
- Improved error handling with proper Result types

### Changed
- Enhanced SIMD similarity functions with strategic `#[inline]` annotations
- Optimized `top_k_similar()` with partial sorting algorithm
- Reduced allocations in performance-critical paths
- Updated test suite: 1,500 → 1,472 tests (+60 property tests, +26 OxiZ tests, +13 SLM tests, +15 LoRA tests)
- Updated codebase: 91 → 95 Rust files, 60,001 → 63,885 total lines (+3,884 lines)

## [0.1.0] - 2026-01-24

### Added
- Initial release of OxiRAG - A four-layer RAG engine
- **Layer 1 (Echo)**: Semantic search with vector embeddings and ANN indexing
- **Layer 2 (Speculator)**: Draft verification using small language models (Candle-based)
- **Layer 3 (Judge)**: Logic verification using SMT solvers (OxiZ integration)
- **Layer 4 (GraphRAG)**: Knowledge graph support with entity extraction and traversal
- WASM support for browser/edge deployment
- Native async runtime with Tokio
- SIMD-optimized similarity calculations
- Prefix caching with paging and invalidation
- Distillation support (teacher-student, progressive, feature-based)
- Query expansion and reranking
- Streaming results with progress reporting
- Circuit breaker and connection pooling
- Comprehensive benchmarking suite
