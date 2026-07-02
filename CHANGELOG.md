# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.21.0] - 2026-07-02

Twelve cutting-edge RAG modules across four themes — Graph-Structured Knowledge RAG, Fine-Grained &
Late-Interaction Retrieval, Tabular & Structured Knowledge, and Reranking/Adaptive-Control/Evaluation
— all pure-Rust heuristic implementations with zero new dependencies. +817 tests (9,306 → 10,123),
173 module directories.

### Added

- **`g_retriever`** (`g-retriever`): G-Retriever (He et al. 2024) — subgraph retrieval framed as a
  **Prize-Collecting Steiner Tree** optimization. Genuine Goemans–Williamson primal-dual moat-growing
  approximation (union-find cluster merges + prize-budget deactivation) followed by
  Johnson–Minkoff–Phillips strong pruning; supports rooted/unrooted modes, edge prizes via the
  virtual-midpoint reduction, and a size cap. `GRetrieverEngine`, `PcstSolver`, `GRetrieverSubgraph`.
  Distinct from `knowledge_graph_qa`'s plain BFS expansion. 57 tests.
- **`think_on_graph`** (`think-on-graph`): Think-on-Graph / ToG (Sun et al. 2023) — LLM-guided
  bounded-width **beam search over KG reasoning paths**: relation exploration → entity exploration →
  relevance prune to width `W` → coverage-based sufficiency check with early stop, over its own
  self-contained graph. `TogEngine`, `TogBeamPath`, `TogKnowledgeGraph`. Distinct from `multi_hop`'s
  exhaustive unpruned edge-following. 59 tests.
- **`lightrag`** (`lightrag`): LightRAG (Guo et al. 2024) — **dual-keyword-type** retrieval (low-level
  entity/relation keys vs high-level theme/concept keys) driving Local/Global/Hybrid paths over an
  incrementally-deduped graph+vector hybrid index (entities merged by canonical key, relations by
  unordered endpoint pair). `LightRagEngine`, `LightRagIndex`, `LightRagDualKeywords`. Distinct from
  `drift_search`'s community-hierarchy coarse→fine (explicitly not community-based). 67 tests.
- **`coil_retrieval`** (`coil-retrieval`): COIL (Gao et al. 2021) — **contextualized inverted lists**:
  postings keyed by surface token, a query token scores a doc only where the exact surface token also
  occurs (exact-lexical gating) via a max per-occurrence contextualized-embedding dot-product, plus an
  optional CLS semantic term (COIL-tok/COIL-full). `CoilRetriever`, `CoilInvertedIndex`, `CoilPosting`.
  Distinct from `plaid_retrieval`'s soft all-to-all MaxSim and `sparse_retrieval`'s SPLADE expansion.
  59 tests.
- **`muvera`** (`muvera`): MUVERA (Dhulipala et al. 2024) — reduces **multi-vector retrieval to
  single-vector MIPS** via Fixed Dimensional Encodings: SimHash space partition, per-cluster
  query-sum/document-average aggregation with Hamming-nearest empty-cell fill, ±1 JL inner projection,
  and repetition concatenation, so one dot-product approximates Chamfer/MaxSim. `MuveraEncoder`,
  `MuveraIndex`, `FixedDimEncoding`. Distinct from `plaid_retrieval` (keeps MaxSim) and `matryoshka`
  (nested dims). 68 tests.
- **`instruction_embed`** (`instruction-embed`): Instruction-conditioned embeddings (INSTRUCTOR/TART,
  Su et al. 2023) — prepends a task instruction to query and document before embedding
  (`combined = base ⊙ gate + shift`, instruction-seeded), so one corpus ranks differently per task
  intent; instruction registry + instruction-aware index. `InstructionEmbedder`, `InstructionIndex`,
  `TaskInstruction`. Distinct from `self_query` (metadata parse) and `matryoshka` (nested dims). 99 tests.
- **`table_rag`** (`table-rag`): TableRAG (Chen et al. 2024) — two-stage retrieval over large tables:
  **schema retrieval** (query vs column names/types) + **cell retrieval** (query expanded into
  (column, cell-value) probes against a capped distinct-value dictionary that bounds cost independent
  of row count), assembling a compact provenance-tracked sub-table. `TableRagEngine`, `TableSchema`,
  `CellProbe`. Distinct from `structured_extraction`'s text→records direction. 70 tests.
- **`structrag`** (`structrag`): StructRAG (Li et al. 2024) — **infers the optimal knowledge structure**
  (Table/Graph/Tree/Catalogue/Algorithm) for a task via a router, **restructures** retrieved passages
  into a genuinely-populated structure of that kind, then reasons over it. `StructRagRouter`,
  `StructRagRestructurer`, `StructRagEngine`, `StructRagKnowledgeStructure`. Distinct from
  `structured_extraction`'s fixed-schema extraction. 85 tests.
- **`chain_of_table`** (`chain-of-table`): Chain-of-Table (Wang et al. 2024) — tabular reasoning by
  planning + executing a **chain of symbolic table-transformation operations** (`f_add_column`,
  `f_select_row`/`f_select_column`, `f_group_by`, `f_sort_by`, `f_aggregate`) that evolve an in-memory
  relational table state until the answer is extractable. `ChainOfTableEngine`, `CotTableOperation`,
  `CotTableState`. Distinct from `program_of_thought`'s arithmetic-scalar DSL. 82 tests.
- **`setwise_rerank`** (`setwise-rerank`): Setwise reranking (Zhuang et al. 2024) — uses a **k-way set
  comparison** as the sort primitive inside heapsort/bubblesort, cutting comparison count from pairwise
  O(n²) toward O(n log n) (the k=2 full sort provably matches the pairwise `n(n-1)/2` bound).
  `SetwiseReranker`, `SetwiseComparison`, `SetwiseSortStrategy`. Distinct from `pairwise_rerank`
  (round-robin tournament) and `listwise_rerank` (RankGPT window permute). 52 tests.
- **`skr`** (`skr`): Self-Knowledge guided Retrieval (Wang et al. 2023) — a **memory-based kNN gate**
  deciding retrieve-or-not from a labeled pool of (question, was-answerable-without-retrieval)
  exemplars via similarity-weighted vote, with a pool-maintenance API (FIFO-capped). `SkrGate`,
  `SelfKnowledgePool`, `SkrDecision`. Distinct from `adaptive_rag`'s stateless complexity tiers and
  `self_route`'s answerability routing. 55 tests.
- **`crud_rag`** (`crud-rag`): CRUD-RAG (Lyu et al. 2024) — a RAG evaluation harness partitioned by the
  **Create/Read/Update/Delete** operation taxonomy, each with genuine hand-derived lexical metrics
  (ROUGE-L + BLEU for Create; EM + token-F1 for Read; correction-similarity + error-span detection for
  Update; key-point coverage − redundancy for Delete). `CrudRagHarness`, `CrudOperation`,
  `CrudRagReport`. Distinct from `rgb_eval`'s four-ability taxonomy. 64 tests.

### Theme umbrellas

- `graph-knowledge-rag` = `g-retriever` + `think-on-graph` + `lightrag`
- `late-interaction` = `coil-retrieval` + `muvera` + `instruction-embed`
- `structured-knowledge` = `table-rag` + `structrag` + `chain-of-table`
- `retrieval-control` = `setwise-rerank` + `skr` + `crud-rag`

## [0.20.0] - 2026-07-02

Twelve cutting-edge RAG modules across four themes — Learned Vector Compression, Decoupled &
Persistent Reasoning Architectures, Prompt & Context Efficiency, and Statistical Calibration &
Privacy Robustness — all pure-Rust heuristic implementations with zero new dependencies. +537
tests (8,769 → 9,306), 161 module directories.

### Added

- **`anisotropic_vq`** (`anisotropic-vq`): ScaNN-style Anisotropic Vector Quantization (Guo et al.
  2020) — decomposes quantization residual into components parallel/orthogonal to each vector's own
  direction, reweights the loss (parallel error penalized more), and fits codewords via a weighted
  Lloyd variant whose centroid update solves a per-cluster weighted-least-squares linear system (not
  a simple mean). `AnisotropicQuantizer`, `AnisotropicVqIndex`. Distinct from uniform-MSE
  `product_quantization` and random-rotation `rabitq`. 38 tests.
- **`itq_hashing`** (`itq-hashing`): Iterative Quantization (Gong & Lazebnik 2011) — PCA projection
  followed by an alternating-minimization loop that *learns* an optimal orthogonal rotation via the
  orthogonal Procrustes problem (closed-form via a from-scratch pure-Rust one-sided Jacobi SVD), unlike
  `rabitq`'s fixed random rotation. `ItqHasher`, `ItqIndex`. 60 tests.
- **`residual_vq`** (`residual-vq`): Residual/Multi-stage Vector Quantization — a cascade of `M`
  k-means codebooks where stage *k* encodes the residual left by stages `0..k`, with additive
  lookup-table distance estimation and an optional beam-search encoder. Distinct from
  `product_quantization`'s parallel subspace split. `ResidualQuantizer`, `ResidualVqIndex`. 30 tests.
- **`rewoo`** (`rewoo`): ReWOO decoupled reasoning (Xu et al. 2023) — a single upfront Planner call
  emits a plan with placeholder evidence variables (`#E1`, `#E2`, ...); a Worker resolves each via
  exactly one retrieval call per step (zero interleaving, unlike `agentic`'s ReAct loop); a Solver
  substitutes resolved evidence into the final answer. `RewooPlanner`, `RewooWorker`, `RewooSolver`.
  60 tests.
- **`searchain`** (`searchain`): Search-in-the-Chain (Xu et al. 2024) — builds a complete global
  reasoning chain upfront, then runs an Interactive-Reasoning-Verification pass that marks each node
  Verified/Unverified/Conflicting and **backtracks** (with transitive multi-level cascade) to
  re-verify dependents when an earlier node is invalidated. `SearChainEngine`, `NodeVerdict`. Distinct
  from `knowledge_conflict` (no chain/backtrack) and `query_planning` (no verify-then-backtrack pass).
  35 tests.
- **`buffer_of_thoughts`** (`buffer-of-thoughts`): Buffer of Thoughts (Yang et al. 2024) — a
  persistent, growing meta-buffer of distilled "thought templates" retrieved by problem-structure
  similarity, instantiated for new problems, and replenished by distilling successful solutions back
  in. `ThoughtBuffer`, `BotEngine`. Distinct from `self_discover`'s static module bank. 70 tests.
- **`llmlingua`** (`llmlingua`): LLMLingua-style prompt compression (Jiang et al. 2023) — a self-fit,
  properly-normalized interpolated n-gram surrogate language model drives coarse (segment-level) then
  fine (token-level) perplexity-based pruning, with a proportional water-filling `BudgetController`
  allocating retention by information density. `PerplexityCompressor`, `PerplexityModel`. 30 tests.
- **`memory_paging`** (`memory-paging`): MemGPT-style virtual context management (Packer et al. 2023)
  — OS-inspired paging between a bounded `MainContext` and unbounded `ArchivalStore`, with
  function-call-triggered page-in/page-out and self-directed LRU/importance eviction under context
  pressure. `ContextPager`, `MemoryPage`. Distinct from `long_term_memory` (no paging) and
  `memory_compression` (recursive summarization, no page-in/page-out). 63 tests.
- **`uprise_retrieval`** (`uprise-retrieval`): UPRISE-style universal prompt retrieval (Cheng et al.
  2023) — retrieves reusable few-shot exemplars from a cross-task pool via embedding similarity,
  reranked by an outcome-quality signal updated via exponential moving average. `UpriseIndex`,
  `UpriseRetriever`. 48 tests.
- **`conformal_rag`** (`conformal-rag`): Split conformal prediction for calibrated RAG confidence —
  a distribution-free finite-sample quantile threshold (`⌈(n+1)(1−α)⌉`-th order statistic, with a
  float-precision snap guard) over calibration-set nonconformity scores, yielding prediction sets with
  a proven marginal coverage guarantee. `ConformalCalibrator`, `MondrianConformalCalibrator`. Distinct
  from `abstention`'s heuristic confidence-margin threshold. 22 tests.
- **`chainpoll`** (`chainpoll`): ChainPoll hallucination detection (Friel & Sanyal 2023) — polls a
  claim's grounding across a fixed, deterministic bank of structurally distinct prompt formulations
  (including inverted framings) and majority-votes, calibrated so unanimous/narrow votes map to
  extreme/mid scores. `ChainPollScorer`. Distinct from `selfcheckgpt`'s stochastic-sampling
  consistency. 41 tests.
- **`membership_inference`** (`membership-inference`): Canary-based membership-inference privacy audit
  for RAG corpora — injects member/non-member canary documents, runs shadow queries, and computes a
  genuine rank-based (Mann-Whitney U) AUC separability score between the two groups, plus a
  verbatim-redaction/confidence-quantization defense. `CanaryAuditor`, `MembershipDefense`. 40 tests.
- **Theme umbrellas**: `advanced-quantization`, `reasoning-architectures`, `context-efficiency`,
  `robust-eval`.

## [0.19.0] - 2026-07-02

Twelve cutting-edge RAG modules across four themes — Scalable Indexing & Quantization, Query
Transformation & Clarification, Retrieval-Augmented Reasoning & Compression, and Hallucination
Detection & Trustworthy Eval — all pure-Rust heuristic implementations with zero new dependencies.
+1,015 tests (7,754 → 8,769), 149 module directories.

### Added

- **`disk_ann`** (`disk-ann`): DiskANN/Vamana graph ANN index (Subramanya et al. 2019) — single-layer
  graph with medoid entry point, `RobustPrune(α)` occlusion-based edge selection, two-pass build
  (α=1.0 then configured α), bounded out-degree, beam-search greedy traversal. `VamanaGraph`,
  `DiskAnnIndex`. Distinct from multi-layer `hnsw_index`. 80 tests.
- **`spann`** (`spann`): SPANN memory-disk hybrid ANN (Chen et al. 2021) — balanced k-means clustering
  with posting-length-limited cluster splitting, boundary-closure replication, and RNG-rule replica
  pruning. `SpannIndex`, `Posting`. Distinct from plain `ivf_index`. 77 tests.
- **`rabitq`** (`rabitq`): RaBitQ randomized-rotation 1-bit quantization (Gao & Long 2024) — FNV-seeded
  deterministic orthogonal rotation, hypercube-vertex codebook, unbiased inner-product/L2 distance
  estimator with a documented error bound. `RaBitQuantizer`, `RaBitQIndex`. Distinct from
  `scalar_quantization`/`product_quantization`. 82 tests.
- **`rq_rag`** (`rq-rag`): RQ-RAG learn-to-refine query router (Chan et al. 2024) — classifies a query
  into Rewrite/Decompose/Disambiguate/Respond and dispatches to the matching refinement strategy.
  `QueryRefinementEngine`, `RefinementPlan`. 100 tests.
- **`tree_of_clarifications`** (`tree-of-clarifications`): Tree of Clarifications (Kim et al. 2023) —
  builds a disambiguation tree for ambiguous questions, retrieves/answers each branch, recursively
  prunes low-relevance branches, aggregates surviving leaves into a long-form answer. `ToCEngine`,
  `ClarificationTree`. 87 tests.
- **`query2doc`** (`query2doc`): Query2Doc pseudo-document expansion (Wang et al. 2023) — generates a
  hypothetical passage and concatenates a repetition-weighted original query with it for sparse/dense
  retrieval. `Query2DocExpander`, `PseudoDocument`. Distinct from HyDE and index-time `doc2query`.
  72 tests.
- **`retrieval_augmented_thoughts`** (`retrieval-augmented-thoughts`): RAT (Wang et al. 2024) —
  generates an initial chain-of-thought, then revises each thought step left-to-right conditioned on
  retrieval targeted at that step. `RatEngine`, `RatTrace`. Distinct from whole-answer
  `iterative_rag`. 82 tests.
- **`recomp`** (`recomp`): RECOMP dual context compressors (Xu et al. 2023) — extractive sentence
  selection and template-based abstractive-lite summarization, gated by a selective-augmentation
  relevance check that can skip low-value retrieval. `RecompPipeline`, `RecompDecision`. 84 tests.
- **`filco`** (`filco`): FILCO sentence-granularity content filtering (Wang et al. 2023) — three
  measures (STRINC string-inclusion, lexical overlap, CXMI-lite conditional utility) to keep only
  generation-useful sentences. `FilcoFilter`, `FilterMeasure`. Distinct from passage-level
  `noise_filter`. 84 tests.
- **`selfcheckgpt`** (`selfcheckgpt`): SelfCheckGPT zero-resource hallucination detection (Manakul et
  al. 2023) — n-gram, NLI-lite, and QA-lite sampling-consistency variants scoring each sentence's
  corroboration across K stochastic samples. `SelfCheckScorer`, `SelfCheckVariant`. 78 tests.
- **`erag`** (`erag`): eRAG retriever evaluation via per-document downstream utility (Salemi & Zamani
  2024) — runs the downstream task on each retrieved document individually, aggregates utility, and
  correlates (Kendall's τ, Spearman's ρ) with end-to-end quality across a case batch. `ERagEvaluator`,
  `kendall_tau`, `spearman_rho`. 79 tests.
- **`eigenscore`** (`eigenscore`): INSIDE/EigenScore hallucination detection (Chen et al. 2024) — a
  pure-Rust cyclic Jacobi eigensolver computes the differential entropy of the regularized covariance
  of K sampled-response embeddings. `EigenScoreDetector`, `symmetric_eigenvalues`. 110 tests.
- **Theme umbrellas**: `scalable-indexing`, `query-transformation`, `reasoning-compression`,
  `trust-eval`.

### Fixed

- **`advanced_retrieval::mmr`**: fixed a stale doctest passing an owned `Vec<SearchResult>` where
  `MmrReranker::rerank` expects `&[SearchResult]`.
- **`retrieval_loop::generator`**: fixed a doctest using `gen` as a local variable name, which became a
  reserved keyword under edition 2024.
- **`chunking::strategies`**: fixed three stale doctests (`SentenceChunker`, `RecursiveChunker`,
  `MarkdownChunker`) that omitted `.with_min_chunk_size(1)`, causing short example text to be filtered
  below the default `min_chunk_size` of 50 and produce zero chunks.

## [0.18.0] - 2026-06-14

Twelve cutting-edge RAG modules across four themes — ANN & Vector Indexing, Advanced Prompting & Reasoning,
Reranking & Retrieval Quality, and Evaluation & Safety — all pure-Rust heuristic implementations with zero
new dependencies. +809 tests (6,945 → 7,754), 137 module directories.

### Added

- **`hnsw_index`** (`hnsw`): Hierarchical Navigable Small World graph ANN index with deterministic level
  assignment, multi-layer greedy search, and `ef_construction`/`ef_search` beam search.
- **`lsh_index`** (`lsh`): Locality-Sensitive Hashing with random-hyperplane LSH (cosine) and banded
  MinHash (Jaccard); deterministic hyperplane generation via FNV-1a.
- **`scalar_quantization`** (`scalar-quantization`): int8 and binary scalar quantization with calibration,
  encode/decode, asymmetric dot product, and Hamming distance.
- **`step_back`** (`step-back`): Step-Back Prompting — abstract a query to a higher-level question,
  retrieve on it, then synthesize the answer.
- **`least_to_most`** (`least-to-most`): Least-to-Most decomposition — split complex queries into ordered
  sub-problems, solve sequentially with prior answers as context.
- **`self_discover`** (`self-discover`): Self-Discover — SELECT/ADAPT/IMPLEMENT atomic reasoning modules
  into a structured plan; includes built-in bank of 8 reasoning modules.
- **`pairwise_rerank`** (`pairwise-rerank`): Tournament-style pairwise reranking (monoT5/duoT5-inspired)
  accumulating wins across rounds; `PairwiseReranker`.
- **`multi_query`** (`multi-query`): Multi-Query expansion — generate N query variants, retrieve for each,
  fuse with RRF; `MultiQueryGenerator`.
- **`citation_verification`** (`citation-verification`): Verify claimed spans are grounded in source
  passages via token overlap and n-gram substring matching; `CitationVerifier`, `VerifiedCitation`.
- **`faithfulness_eval`** (`faithfulness-eval`): RAGAS-style faithfulness scoring — decompose answer into
  atomic claims, entail each against context, aggregate; `FaithfulnessEvaluator`.
- **`prompt_injection_defense`** (`prompt-injection-defense`): Detect injection/jailbreak text in retrieved
  context with 40+ patterns; configurable Quarantine/Sanitize/Flag/ScoreOnly strategies;
  `PromptInjectionDetector`.
- **`query_difficulty`** (`query-difficulty`): Predict query difficulty band (Easy/Medium/Hard/Ambiguous)
  from negation, multi-hop, ambiguity, rarity, and length signals; `DifficultyPredictor`.
- **Theme umbrellas**: `vector-indexing`, `advanced-prompting`, `rerank-quality`, `eval-safety`.

## [0.17.0] - 2026-06-14

Twelve more cutting-edge RAG technique modules across four themes — ANN Indexing & Late Interaction,
Generation Refinement, Robustness & Privacy, and Advanced Evaluation — all pure-Rust heuristic
implementations with zero new dependencies.

### Added

#### Theme 1 — ANN Indexing & Late Interaction

- **Product Quantization** (feature `product-quantization`): Jégou et al. 2011. Splits vectors into subspaces, learns per-subspace codebooks via deterministic k-means-lite, encodes as codeword indices, and computes Asymmetric Distance (ADC) via lookup tables. `ProductQuantizer`, `PqIndex`. Distinct from scalar `quantization` (INT8/INT4). 71 tests.
- **IVF Index** (feature `ivf-index`): inverted-file ANN. Coarse-quantizer centroids + inverted lists; search probes the `nprobe` nearest cells. `IvfIndex.build()/search()`. Distinct from HNSW graph ANN. 54 tests.
- **PLAID** (feature `plaid`): ColBERTv2/PLAID late interaction (Santhanam et al. 2022). Per-token embeddings + MaxSim scoring with centroid-based candidate generation/pruning. `PlaidRetriever`. Distinct from basic `multi_vector`. 65 tests.

#### Theme 2 — Generation Refinement

- **Self-Refine** (feature `self-refine`): Madaan et al. 2023. Iterative self-feedback → refine loop with the same model (no external memory/retrieval, unlike Reflexion). `SelfRefineEngine.run`. 70 tests.
- **Chain-of-Density** (feature `chain-of-density`): Adams et al. 2023. Iteratively densifies a fixed-length summary by incorporating salient missing entities while trimming filler. `ChainOfDensityEngine.summarize`, `DensityStep`. 65 tests.
- **Analogical Prompting** (feature `analogical`): Yasunaga et al. 2023. Self-generates relevant exemplars + high-level knowledge before solving. `AnalogicalEngine.run`. Distinct from `prompt_optimization` (which selects from a pool). 56 tests.

#### Theme 3 — Robustness & Privacy

- **Poisoning Defense** (feature `poisoning-defense`): RAG-security detector for adversarial/poisoned passages — keyword-stuffing density, lexical-diversity (type/token), and consensus-anomaly signals combine into a poison risk. `PoisoningDetector.scan()/filter()`. Distinct from `noise_filter` (irrelevant distractors). 66 tests.
- **Anonymization** (feature `anonymization`): PII pseudonymization via deterministic char-class scanning (emails, phones, person names) → consistent placeholders with a reversible mapping; `Anonymizer.anonymize()/deanonymize()`. Distinct from `guardrails` (irreversible redaction). 54 tests.
- **Abstention** (feature `abstention`): selective prediction — answer-or-refuse from confidence + retrieval support, plus risk-coverage curve analysis. `AbstentionPolicy.assess()/risk_coverage()`. 60 tests.

#### Theme 4 — Advanced Evaluation

- **RAGChecker** (feature `ragchecker`): Ru et al. 2024. Claim-level diagnostics — retriever (claim recall, context precision) + generator (faithfulness, hallucination rate, correctness, noise sensitivity) via claim entailment. `RagChecker.check()`. 54 tests.
- **Retrieval Diversity** (feature `retrieval-diversity`): diversity/coverage metrics — intra-list diversity, subtopic recall (S-recall), and novelty-discounted α-nDCG. `DiversityScorer.compute()`. 60 tests.
- **ARES Eval** (feature `ares-eval`): Saad-Falcon et al. 2023. Automated eval with prediction-powered inference (PPI) — debiases a judge's mean on unlabeled data by its measured bias on a labeled set, yielding a tighter confidence interval. `AresEvaluator.ppi_estimate()`, `PpiInterval`. 52 tests.

- **Four v0.17.0 theme umbrella features**: `ann-indexing`, `generation-refinement`, `robustness-privacy`, `advanced-eval`.

### Codebase Statistics (v0.17.0)

- **Tests**: 6,945 (all passing; +727 from v0.16.0's 6,218)
- **Clippy Warnings**: 0 (`--all-features --all-targets -D warnings`)
- **Rustdoc Warnings**: 0
- **New features**: `product-quantization`, `ivf-index`, `plaid`, `self-refine`, `chain-of-density`, `analogical`, `poisoning-defense`, `anonymization`, `abstention`, `ragchecker`, `retrieval-diversity`, `ares-eval`, `ann-indexing`, `generation-refinement`, `robustness-privacy`, `advanced-eval`
- **New modules**: `src/product_quantization/`, `src/ivf_index/`, `src/plaid_retrieval/`, `src/self_refine/`, `src/chain_of_density/`, `src/analogical_prompting/`, `src/poisoning_defense/`, `src/anonymization/`, `src/abstention/`, `src/ragchecker/`, `src/retrieval_diversity/`, `src/ares_eval/`

## [0.16.0] - 2026-06-14

Twelve more cutting-edge RAG technique modules across four themes — Graph & Generative Retrieval,
Retrieval Composition, Time/Language/Personalization, and Grounding & Fine-grained Verification —
all pure-Rust heuristic implementations with zero new dependencies.

### Added

#### Theme 1 — Graph & Generative Retrieval

- **DRIFT Search** (feature `drift-search`): GraphRAG DRIFT (Microsoft 2024). Combines global community-summary search with local entity search via iterative follow-up sub-queries that drill down the community hierarchy. Self-contained `DriftSearchEngine.search()` over caller-supplied `CommunityReport`s. 54 tests.
- **Entity Linking** (feature `entity-linking`): mention → canonical KB entity disambiguation. `EntityCatalog` with a case-insensitive alias index; `EntityLinker.link()` resolves ambiguous mentions by context-vs-description cosine, emitting NIL below a confidence floor. 65 tests.
- **Generative Retrieval** (feature `generative-retrieval`): DSI-style (Tay et al. 2022). Assigns each doc a hierarchical semantic doc-id (cluster path) and "generates" the id via beam-search constrained traversal of the centroid tree. `GenerativeRetriever`, `SemanticDocId`. 68 tests.

#### Theme 2 — Retrieval Composition

- **Auto-Merging** (feature `auto-merging`): LlamaIndex auto-merging. Collapses retrieved child chunks into their parent when ≥ a threshold fraction of siblings are hit (recursively). `AutoMergingRetriever`, `ChunkHierarchy`. Distinct from `parent_document` (one-chunk expansion). 61 tests.
- **Ensemble Retriever** (feature `ensemble-retriever`): orchestrates multiple pluggable `SubRetriever` traits with weights and fuses via `WeightedScore` or `WeightedRrf` — a doc surfaced by several retrievers ranks higher. Distinct from `rank_fusion` (which fuses given lists). 59 tests.
- **GenRead** (feature `gen-read`): generate-then-read (Yu et al. 2023). Generates diverse contextual documents, clusters them, and uses one representative per cluster as the reading context (no retrieval). `GenReadEngine.run`. Distinct from HyDE. 51 tests.

#### Theme 3 — Time, Language & Personalization

- **Fresh Retrieval** (feature `fresh-retrieval`): FreshLLM-style (Vu et al. 2023). Classifies query time-sensitivity (Static/SlowChanging/FastChanging), scores document freshness by exponential age decay, flags stale answers, and reranks weighted by freshness demand. `FreshnessAnalyzer`. Distinct from `temporal` (unconditional decay). 69 tests.
- **Cross-Lingual** (feature `cross-lingual`): cross-lingual retrieval via a caller-supplied `BilingualLexicon` (query-token translation) + diacritic normalization to a shared form. `CrossLingualRetriever.search()`. 74 tests.
- **Personalized RAG** (feature `personalized-rag`): user-profile-aware reranking. `UserProfile` (topic interests + history) drives an affinity score blended with relevance. `PersonalizedReranker.rerank()`. Distinct from `relevance_feedback` (query-level Rocchio). 55 tests.

#### Theme 4 — Grounding & Fine-grained Verification

- **Quote Grounding** (feature `quote-grounding`): extracts the minimal verbatim supporting quote per answer claim (the best-overlap source sentence, trimmed to a token window). `QuoteGrounder.ground()` / `ungrounded()`. Distinct from `attribution` (citation alignment). 54 tests.
- **Claim Decomposition** (feature `claim-decomposition`): FActScore-style (Min et al. 2023) decomposition of an answer into atomic, decontextualized claims (clause splitting + pronoun→subject resolution). `AtomicClaimExtractor.decompose()`. Distinct from `hallucination_detector` (claim scoring). 56 tests.
- **Fusion-in-Decoder** (feature `fusion-in-decoder`): FiD-style (Izacard & Grave 2021). Extracts per-passage evidence independently then fuses across passages weighted by relevance (deduplicated) into an answer with multi-passage attribution. `FusionInDecoder.fuse()`. 56 tests.

- **Four v0.16.0 theme umbrella features**: `graph-generative`, `retrieval-composition`, `time-lang-personal`, `grounding-verification`.

### Codebase Statistics (v0.16.0)

- **Tests**: 6,218 (all passing; +722 from v0.15.0's 5,496)
- **Clippy Warnings**: 0 (`--all-features --all-targets -D warnings`)
- **Rustdoc Warnings**: 0
- **New features**: `drift-search`, `entity-linking`, `generative-retrieval`, `auto-merging`, `ensemble-retriever`, `gen-read`, `fresh-retrieval`, `cross-lingual`, `personalized-rag`, `quote-grounding`, `claim-decomposition`, `fusion-in-decoder`, `graph-generative`, `retrieval-composition`, `time-lang-personal`, `grounding-verification`
- **New modules**: `src/drift_search/`, `src/entity_linking/`, `src/generative_retrieval/`, `src/auto_merging/`, `src/ensemble_retriever/`, `src/gen_read/`, `src/fresh_retrieval/`, `src/cross_lingual/`, `src/personalized_rag/`, `src/quote_grounding/`, `src/claim_decomposition/`, `src/fusion_in_decoder/`

## [0.15.0] - 2026-06-14

Twelve more cutting-edge RAG technique modules (mostly 2023-2024 papers) across four themes —
Next-Gen Retrieval Architectures, Adaptive Generation Strategies, Knowledge & Context Management,
and Evaluation & Benchmarking — all pure-Rust heuristic implementations with zero new dependencies.

### Added

#### Theme 1 — Next-Gen Retrieval Architectures

- **HippoRAG** (feature `hipporag`): Gutiérrez et al. 2024. Builds an entity graph (entities co-occurring in a passage are linked) and runs **Personalized PageRank** seeded from the query's entities to score passages by their entities' PPR mass — capturing multi-hop associations in a *single* retrieval step. `HippoRagIndex.build()/search()`. Distinct from `multi_hop` (BFS). 55 tests.
- **LongRAG** (feature `long-rag`): Jiang et al. 2024. Groups chunks/documents into a small number of *long* retrieval units (`ByDocument` / `BySemanticAdjacency` / `FixedTokenWindow`) up to a token budget, reducing fragmentation. `LongUnitGrouper.group()`, `LongRagRetriever.search()`. 53 tests.
- **DRAGIN** (feature `dragin`): Su et al. 2024. Dynamic retrieval driven by real-time information need — RIND triggers retrieval on high token uncertainty + content-ness; QFS formulates the query from salient tokens of the recent generated window. `DraginEngine.run` over `UncertaintyGenerator`/`Retriever` traits. 65 tests.

#### Theme 2 — Adaptive Generation Strategies

- **Astute RAG** (feature `astute-rag`): Wang et al. 2024. Reconciles internal (parametric) vs external (retrieved) knowledge under imperfect retrieval — extracts external statements with corroboration-based reliability, detects conflicts, resolves by reliability + `prefer_external`, synthesizes an attributed answer. `AstuteConsolidator.consolidate()/run()`. 70 tests.
- **Self-Route** (feature `self-route`): Li et al. 2024. Routes between cheap RAG and long-context based on answerability (query-coverage + top relevance of the retrieved context); falls back to `LongContext` when retrieval is insufficient. `SelfRouter.route()`. Distinct from `adaptive_rag` (complexity→depth). 65 tests.
- **Speculative Drafting** (feature `speculative-drafting`): Wang et al. 2024 Speculative RAG. Clusters retrieved docs (deterministic k-means-lite), drafts one answer per cluster in parallel (`std::thread::scope`), then scores each by verifier support + cross-draft self-consistency and picks the best. `SpeculativeDrafter.run`. 66 tests.

#### Theme 3 — Knowledge & Context Management

- **MemoRAG** (feature `memorag`): Qian et al. 2024. Forms a global corpus memory gist (salient sentences + key terms), generates retrieval *clues* (query ⊕ gist key terms), retrieves evidence per clue and fuses by max-per-doc. `MemoRagEngine.build_memory()/generate_clues()/retrieve()`. 60 tests.
- **Context Pruning** (feature `context-pruning`): LLMLingua-style token-level compression — scores each token by rarity × content-ness × query-overlap (entities optionally preserved), keeps the top tokens to a target ratio while preserving original order. `TokenPruner.prune()`. Distinct from sentence-extractive `context_compression`. 61 tests.
- **Knowledge Conflict** (feature `knowledge-conflict`): inter-passage contradiction detection (negation/numeric/temporal mismatch on a shared subject) and resolution by `Recency` / `Authority` / `Majority` policy. `ConflictDetector.detect()`, `ConflictResolver.resolve()`. Distinct from `consistency_checker` (intra-answer). 61 tests.

#### Theme 4 — Evaluation & Benchmarking

- **RGB Eval** (feature `rgb-eval`): Chen et al. 2023. Evaluates four RAG abilities — noise robustness, negative rejection, information integration, counterfactual robustness — via labeled test cases and lexical correctness/rejection heuristics. `RgbEvaluator.evaluate()`. 54 tests.
- **Nugget Eval** (feature `nugget-eval`): TREC-style nugget scoring — decomposes a reference answer into Vital/Okay information nuggets and scores a system answer by weighted nugget coverage. `NuggetScorer.score()`, `HeuristicNuggetExtractor`. 71 tests.
- **A/B Eval** (feature `ab-eval`): paired A/B pipeline comparison — win/loss/tie counts plus a fully deterministic **paired bootstrap** confidence interval + two-sided p-value (FNV-1a resampling, no `rand`). `AbEvaluator.compare()`. 62 tests.

- **Four v0.15.0 theme umbrella features**: `next-gen-retrieval`, `adaptive-generation`, `knowledge-context`, `eval-benchmarking`.

### Codebase Statistics (v0.15.0)

- **Tests**: 5,496 (all passing; +743 from v0.14.0's 4,753)
- **Clippy Warnings**: 0 (`--all-features --all-targets -D warnings`)
- **Rustdoc Warnings**: 0
- **New features**: `hipporag`, `long-rag`, `dragin`, `astute-rag`, `self-route`, `speculative-drafting`, `memorag`, `context-pruning`, `knowledge-conflict`, `rgb-eval`, `nugget-eval`, `ab-eval`, `next-gen-retrieval`, `adaptive-generation`, `knowledge-context`, `eval-benchmarking`
- **New modules**: `src/hippo_rag/`, `src/long_rag/`, `src/dragin/`, `src/astute_rag/`, `src/self_route/`, `src/speculative_drafting/`, `src/memorag/`, `src/context_pruning/`, `src/knowledge_conflict/`, `src/rgb_eval/`, `src/nugget_eval/`, `src/ab_eval/`

## [0.14.0] - 2026-06-14

Twelve more cutting-edge RAG technique modules across four themes — Retrieval & Indexing,
Ranking & Fusion, Structured Reasoning, and Verification & Robustness — all pure-Rust heuristic
implementations with zero new dependencies.

### Added

#### Theme 1 — Retrieval & Indexing

- **Sparse Retrieval** (feature `sparse-retrieval`): SPLADE-style learned sparse retrieval. `SparseEncoder.fit()` computes IDF + a token co-occurrence table; `encode()` produces a `SparseVector` of log-saturated `ln(1 + max(0, tf*idf))` weights, then expands the term set with discounted co-occurring terms (simulating SPLADE expansion). `SparseIndex.search()` ranks by sparse dot product. Distinct from BM25 (`hybrid_search`). 56 tests.
- **Self-Query** (feature `self-query`): self-querying retriever. `SelfQueryParser.parse()` turns a natural-language query into a structured `ParsedFilter` (FilterCondition with Eq/Ne/Gt/Lt/Gte/Lte/Contains) + a residual semantic query, against a configurable `FilterSchema`. Detects "after/before <year>", "by <author>", "above/below <N>", "tagged <X>". `SelfQueryRetriever` applies the filter to document metadata. 65 tests.
- **Summary Index** (feature `summary-index`): LlamaIndex `DocumentSummaryIndex`. `ExtractiveSummarizer` builds a top-N TF-centrality summary per document; `SummaryIndex.search()` matches the query against summary embeddings but returns the FULL parent document. Flat (one summary per doc), distinct from RAPTOR's recursive tree. 58 tests.

#### Theme 2 — Ranking & Fusion

- **Rank Fusion** (feature `rank-fusion`): multi-list fusion beyond RRF. `RankFusion.fuse()` supports `CombSum`, `CombMnz` (rewards multi-list agreement), `CombAnz`, `Borda`, `Isr`, `WeightedSum`, and `Rrf`, with `ScoreNormalization` (MinMax/ZScore/SumTo1/None). Dedupes by `DocumentId`, deterministic tie-breaking. 55 tests.
- **Diversity Rank** (feature `diversity-rank`): Determinantal-Point-Process-inspired diverse selection. `DiversityRanker.select()` greedily maximizes a volume-style gain `quality² · Π(1 − sim)` over the selected set — distinct from MMR's linear λ trade-off — with a quality-fill fallback when no positive-gain candidate remains. 60 tests.
- **Source Credibility** (feature `source-credibility`): source authority scoring (distinct from answer-level `trust_score`). `SourceGraph.pagerank()` runs iterative PageRank over a citation graph (dangling-mass redistribution); `CredibilityScorer.score()` blends PageRank + exponential recency decay + metadata authority (trusted sources, author reputation); `rerank()` blends relevance with credibility. 75 tests.

#### Theme 3 — Structured Reasoning

- **Graph-of-Thoughts** (feature `graph-of-thought`): Besta et al. 2023. Generalizes ToT to a DAG of thoughts with `GotOperation::{Generate, Aggregate, Refine, KeepBest}`. `GraphOfThoughtEngine.run` executes an operation plan over a frontier using caller-supplied sync `ThoughtGenerator`/`ThoughtScorer`/`ThoughtAggregator` traits (with mocks). 55 tests.
- **Skeleton-of-Thought** (feature `skeleton-of-thought`): Ning et al. 2023. Two-stage generation — produce an answer skeleton (point headers) then expand each point independently, joining into the final answer. `SkeletonOfThoughtEngine.run` over `SkeletonGenerator`/`PointExpander` traits. 48 tests.
- **Program-of-Thoughts** (feature `program-of-thought`): Chen et al. 2022. Separates reasoning from computation via a real arithmetic DSL — `parse_program()` is a recursive-descent parser (standard precedence, parentheses, unary minus) and `Interpreter.eval()` executes assignments/return over a variable environment (div-by-zero & undefined-variable errors). `ProgramOfThoughtEngine.run` generates → parses → evaluates. 74 tests.

#### Theme 4 — Verification & Robustness

- **Fact Check** (feature `fact-check`): FEVER-style verification. `FactChecker.verify()` retrieves evidence sentences and assigns a `Verdict` (Supports / Refutes / NotEnoughInfo) from lexical overlap + contradiction detection (negation mismatch, number conflicts), with confidence + rationale. Distinct from `hallucination_detector` (Jaccard claim-support). 60 tests.
- **Noise Filter** (feature `noise-filter`): RAAT-inspired robustness. `NoiseFilter.filter()` scores each passage's query relevance + consensus alignment (similarity to the set centroid), flags distractors, and drops them while keeping at least `keep_min`. Distinct from redundancy filtering (`context_compression`). 53 tests.
- **Answer Calibration** (feature `answer-calibration`): answer-level confidence calibration + metrics. `AnswerCalibrator.estimate_confidence()` blends verbalized confidence (parses "90%"/hedges), self-agreement, and retrieval support; free functions `compute_metrics`/`expected_calibration_error`/`brier_score` produce ECE/MCE/Brier + a reliability diagram; `fit_platt()` fits a logistic calibration map. 81 tests.

- **Four v0.14.0 theme umbrella features**: `retrieval-indexing`, `ranking-fusion`, `structured-reasoning`, `verification-robustness`.

### Codebase Statistics (v0.14.0)

- **Tests**: 4,753 (all passing; +740 from v0.13.0's 4,013)
- **Clippy Warnings**: 0 (`--all-features --all-targets -D warnings`)
- **Rustdoc Warnings**: 0
- **New features**: `sparse-retrieval`, `self-query`, `summary-index`, `rank-fusion`, `diversity-rank`, `source-credibility`, `graph-of-thought`, `skeleton-of-thought`, `program-of-thought`, `fact-check`, `noise-filter`, `answer-calibration`, `retrieval-indexing`, `ranking-fusion`, `structured-reasoning`, `verification-robustness`
- **New modules**: `src/sparse_retrieval/`, `src/self_query/`, `src/summary_index/`, `src/rank_fusion/`, `src/diversity_rank/`, `src/source_credibility/`, `src/graph_of_thought/`, `src/skeleton_of_thought/`, `src/program_of_thought/`, `src/fact_check/`, `src/noise_filter/`, `src/answer_calibration/`

## [0.13.0] - 2026-06-14

Twelve new cutting-edge RAG technique modules across four themes — Index-Time Representation,
Reranking & Result-Set Selection, Compositional Reasoning, and Calibration/Uncertainty/Geometry —
all pure-Rust heuristic implementations with zero new dependencies.

### Added

#### Theme 1 — Index-Time Representation

- **Late Chunking** (feature `late-chunking`, depends on `chunking`): Jina AI 2024. Embeds the whole document into contextual token vectors *first*, then pools each chunk's token span ("late" pooling) so every chunk embedding carries document context. `LateChunker.encode_document()` blends per-token FNV-1a base embeddings with the document mean by `context_weight`, tiles `chunk_size_tokens` spans with `overlap_tokens`, and pools (`Mean`/`Max`). `naive_chunks()` provides the context-free baseline; `context_weight = 0` makes late ≡ naive. 58 tests.
- **Proposition Retrieval** (feature `proposition-retrieval`): Chen et al. 2023 Dense-X. Decomposes passages into atomic, self-contained propositions via clause splitting + pronoun resolution (`HeuristicPropositionExtractor`), indexes propositions → parent docs (`PropositionIndex`), and retrieves via best-matching proposition. `search()` returns `PropositionHit`s; `search_documents()` dedupes to the best proposition per parent. 59 tests.
- **Doc2Query Expansion** (feature `doc2query`): Nogueira et al. doc2query/HyPE index-time expansion. `HeuristicQueryGenerator` generates hypothetical questions a passage answers (definitional / factoid / relational families over TF-salient terms and capitalized entities). `Doc2QueryExpander.expand()` / `to_indexable_document()` append generated queries to passage text for re-indexing. 48 tests.

#### Theme 2 — Reranking & Result-Set Selection

- **Listwise Reranking** (feature `listwise-rerank`): Sun et al. 2023 RankGPT. Sliding-window listwise permutation reranking — slide a window from the back of the list to the front, permuting each window via `ListwiseJudge` (`LexicalListwiseJudge` scores by window-local IDF + token overlap). `ListwiseReranker.rerank()` returns `ListwiseResult`s with original/new ranks; guards `step = 0` and `window ≥ n`. 55 tests.
- **Autocut** (feature `autocut`): Weaviate-style relevance-gap truncation. `AutoCutter.cut()` detects score discontinuities and dynamically truncates the result list. Four `AutoCutStrategy` variants — `Jumps(k)` (cut after k significant gaps), `RelativeThreshold(r)`, `StdDev(s)`, `Knee` (largest single gap) — clamped to `[min_keep, max_keep]`. `cut_with_report()` returns an `AutoCutReport`. 59 tests.
- **Semantic Dedup** (feature `semantic-dedup`): LSH near-duplicate removal distinct from cosine redundancy filtering. SimHash (64-bit, frequency-weighted, Hamming distance) and MinHash (k-shingles, seeded permutations, Jaccard estimate). `SemanticDeduplicator.deduplicate()` clusters near-duplicates via union-find single-linkage and keeps one representative per `KeepPolicy` (`First`/`HighestScore`/`Longest`). 64 tests.

#### Theme 3 — Compositional Reasoning

- **Self-Ask** (feature `self-ask`): Press et al. 2022. Explicit follow-up decomposition — iteratively emit `Follow up:` sub-questions, answer each via a supplied `SubAnswerer`, and compose the final answer. `SelfAskEngine.run<M, A>` is generic over caller-supplied `SelfAskModel` + `SubAnswerer` traits (with mocks); `SelfAskTrace` records every hop. 52 tests (+1 doctest).
- **Self-Consistency** (feature `self-consistency`): Wang et al. 2022. Samples diverse reasoning paths, clusters final answers by semantic equivalence (normalized-string + token-Jaccard single-linkage), and marginalizes — highest-vote cluster wins, confidence = its vote share. `SelfConsistencyEngine.run<S>` / `marginalize()`; `VoteWeighting::{Uniform, ByReasoningLength}`; deterministic tie-breaking. 67 tests (+1 doctest).
- **Adaptive-RAG** (feature `adaptive-rag`, depends on `advanced-retrieval`): Jeong et al. 2024. Query-complexity classifier (`Straightforward` / `SingleStep` / `MultiStep`) from weighted heuristic signals routes retrieval depth. `AdaptiveRagRouter.route()` returns a `RoutingPlan` (strategy + recommended top-k + max hops). Distinct from `query-routing` (intent → modality). 72 tests.

#### Theme 4 — Calibration, Uncertainty & Embedding Geometry

- **Semantic Entropy** (feature `semantic-entropy`): Kuhn et al. 2023. Clusters sampled answers by bidirectional-entailment (token-Jaccard + mutual containment), computes discrete semantic entropy over the cluster distribution as an uncertainty/hallucination signal. `SemanticEntropyEstimator.estimate()` / `estimate_weighted()` / `is_uncertain()`; `predictive_entropy` baseline ≥ semantic entropy. 66 tests (+1 doctest).
- **Matryoshka** (feature `matryoshka`): Kusupati et al. 2022. Nested truncatable embeddings (decay-weighted so prefixes stay meaningful) for coarse-to-fine retrieval. `MatryoshkaEmbedding.truncate(dim)` renormalizes a prefix; `MatryoshkaRetriever.search()` runs two-stage retrieval (shortlist by `shortlist_dim` prefix, rerank shortlist by full dim). Distinct from `quantization` (precision). 68 tests.
- **Synthetic Eval** (feature `synthetic-eval`): RAGAS/ARES-style test-set generation. `SyntheticEvalGenerator.generate()` produces `SyntheticQa` tuples (question, ground-truth answer, source doc, hard distractors) from a corpus via `HeuristicTemplater` (`Factoid`/`Definitional`/`Cloze`/`Relational`) with lexical-overlap distractor mining. Distinct from the scoring modules. 60 tests.

- **Four v0.13.0 theme umbrella features**: `index-representation`, `rerank-selection`, `compositional-reasoning`, `calibration-geometry`.

### Codebase Statistics (v0.13.0)

- **Tests**: 4,013 (all passing; +728 from v0.12.0's 3,285)
- **Clippy Warnings**: 0 (`--all-features --all-targets -D warnings`)
- **Rustdoc Warnings**: 0
- **New features**: `late-chunking`, `proposition-retrieval`, `doc2query`, `listwise-rerank`, `autocut`, `semantic-dedup`, `self-ask`, `self-consistency`, `adaptive-rag`, `semantic-entropy`, `matryoshka`, `synthetic-eval`, `index-representation`, `rerank-selection`, `compositional-reasoning`, `calibration-geometry`
- **New modules**: `src/late_chunking/`, `src/proposition/`, `src/doc2query/`, `src/listwise_rerank/`, `src/autocut/`, `src/semantic_dedup/`, `src/self_ask/`, `src/self_consistency/`, `src/adaptive_rag/`, `src/semantic_entropy/`, `src/matryoshka/`, `src/synthetic_eval/`

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
