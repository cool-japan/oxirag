# OxiRAG TODO

## v0.23.0 — Constrained & Mutable Search, Probabilistic IR, Collaborative Generation & Privacy-Provenance ✅

**Released**: 2026-07-11 | **Tests**: 11,545 | **Warnings**: 0

Twelve cutting-edge RAG modules, four themes, zero new deps, 197 module dirs. +587 tests from v0.22.0.

Pre-validated by 4 parallel recon passes over the existing 185 modules. Recon **dropped 2 candidates**:
an NSG/MRNG index (near-duplicate — `disk_ann`'s `RobustPrune` at `alpha=1.0` *is* the MRNG rule) and a
standalone RM3 module (near-duplicate of `relevance_feedback.rs` Rocchio + `query_expansion/prf.rs`
TF-PRF; folded into `language_model_retrieval` instead, where it is non-duplicative). Differential
privacy was deferred (needs an RNG; conflicts with the crate's deterministic no-`rand` convention).

**Theme 1 — Constrained & Mutable Search** (`efficient-search`)
- [x] **Filtered Vector Search** (`filtered-vector-search`): ACORN/Filtered-DiskANN predicate-aware graph
  traversal (routes *through* non-matching nodes) + selectivity estimator picking pre-/post-/in-filter.
  Measured: at 1% selectivity, naive post-filter recall@10 collapses to 0.090 (its s·m ceiling) while
  in-filter reaches 0.960, pre-filter 1.000. `FilteredVectorIndex`. 68 tests.
- [x] **Dynamic Pruning** (`dynamic-pruning`): WAND / BlockMax-WAND / MaxScore over a real
  term→posting-list BM25 inverted index. Top-k **bit-identical** to exhaustive (409 vs 528 vs 5,543
  postings — 13.5× fewer). `DynamicPruningIndex`. 44 tests.
- [x] **Index Maintenance** (`index-maintenance`): FreshDiskANN tombstone delete + backward-edge-repair
  insert + bridging/Vamana-refine consolidation. Under 30%+30% churn, naive hard-delete degrades to
  0.875 recall (49 orphans) while tombstone+consolidate holds 0.963 (0 orphans). `MaintainableIndex`. 19 tests.

**Theme 2 — Probabilistic IR & Counterfactual Learning** (`probabilistic-ir`)
- [x] **Language-Model Retrieval** (`language-model-retrieval`): query-likelihood (Dirichlet /
  Jelinek-Mercer / absolute-discounting) + DFR (PL2, DPH) + RM3 relevance-model feedback on the LM.
  Hand-computed scores verified to 1e-12. `LmRetrievalIndex`. 42 tests.
- [x] **Click Model** (`click-model`): PBM / Cascade / DBN fit by EM (log-likelihood proven monotone);
  examination probs → propensities → IPS/SNIPS/doubly-robust debiasing. Raw CTR is fooled by position
  bias; the IPS-debiased ranker recovers the true order. `PositionBasedModel`, etc. 65 tests.
- [x] **Bandit Ranker** (`bandit-ranker`): LinUCB (Sherman-Morrison, matched to a direct inverse within
  8.97e-19) + Thompson (Cholesky posterior) + ε-greedy; measured-sublinear regret; replay OPE.
  Deterministic SplitMix64, no `rand`. `LinUcbRanker`, `ThompsonSamplingRanker`. 80 tests.

**Theme 3 — Ensemble & Collaborative Generation** (`collaborative-generation`)
- [x] **REPLUG** (`replug`): output-distribution ensembling `p(y|q) = Σ λ·p(y|d⊕q)`, a mixture of
  DISTRIBUTIONS not logits (geometric-pool difference is a regression test), + LSR KL retriever feedback.
  Ensemble outranks every individual document. `ReplugEngine`. 53 tests.
- [x] **Mixture-of-Agents** (`mixture-of-agents`): layered proposers → aggregator **synthesizes**
  (coverage-preserving, not selection) → seeds next layer. `MoaEngine`. 75 tests.
- [x] **Chain-of-Agents** (`chain-of-agents`): sequential chunk workers passing an evolving communication
  unit → manager. Ablation proves the unit is load-bearing (zeroing it fails a first↔last-chunk question
  the full chain answers). `CoaEngine`. 36 tests.

**Theme 4 — Privacy, Provenance & Context Governance** (`privacy-provenance`)
- [x] **KV-Cache Compression** (`kv-cache-compression`): H2O / StreamingLLM / SnapKV attention-score
  token eviction over a real SDPA KV tensor. At 25% budget, H2O/SnapKV deviate 0.0048 from full
  attention vs naive-LRU 1.0066 (~210×); sink-vs-middle deletion is a ~25,000× asymmetry.
  `KvCacheCompressor`. 31 tests.
- [x] **Watermarking** (`watermarking`): Kirchenbauer green-list biasing (hash-seeded ⇒ deterministic),
  soft/hard, z-score detection with self-implemented normal CDF. True-positive z=34.6; FPR calibrated to
  the normal tail (0.0537 vs 0.05 at z≥1.645). `WatermarkGenerator`, `WatermarkDetector`. 47 tests.
- [x] **Knowledge Unlearning** (`knowledge-unlearning`): GDPR right-to-be-forgotten — near-dup+artifact
  scope resolution → delete → leakage re-audit → escalate or **honestly report non-convergence**.
  Delete-by-id alone leaks a paraphrase at 0.569; full engine re-audits clean. `UnlearningEngine`. 27 tests.

**Bug fix (surfaced by recon)**
- [x] `layer2_speculator/candle_slm.rs` returned `last_logits.max(0)` (max **raw logit**, unnormalized)
  as a "logprob" — in both `generate_internal` and `get_logprobs`. Fixed via a shared
  `sampled_token_logprob` helper (log-softmax, then the sampled token's value); tests/vectors stay aligned.

**Integration & Release**
- [x] 12 features + 4 umbrellas (`efficient-search`, `probabilistic-ir`, `collaborative-generation`,
  `privacy-provenance`); 12 modules + prelude re-exports, **0 aliases needed** (210 public names cross-
  checked; proactive module-prefixing held — matches the v0.21.0 zero-alias result).
- [x] Toolchain-drift cleanup (toolchain updated 2026-07-07 since v0.22.0): 7 new clippy errors in 5
  files (`manual_assert_eq`/`question_mark`/`useless_borrows_in_formatting`) + 61 rustdoc intra-doc-link
  warnings across 26 files, all fixed at root cause (no `#[allow]`). Added missing `required-features`
  for 3 examples + `trait_bounds_native` test (pre-existing packaging gap).
- [x] `cargo fmt` + build/clippy/nextest/doc `--all-features --all-targets -D warnings` → 0 errors,
  0 warnings, 11,545 tests + 192 doctests green.
- [x] Bumped 0.22.0 → 0.23.0; updated CHANGELOG.md, TODO.md.
- [x] Process note: entered via `/ucont` on Opus (Rule 0 satisfied — orchestrator ran on Opus 4.8 this
  cycle, unlike v0.19–v0.22 which fell back to Sonnet). Recon-first paid off again (2 duplicates dropped
  pre-coding). Zero git-reset hazard (no-git-mutation briefs held) and zero cross-module collisions (each
  of 12 impl agents owned exactly one dir; Phase-0 pre-wired all shared files). All 6 opus modules caught
  real bugs their own measurement-based tests exposed — bugs that assertion-only tests would have shipped:
  dynamic_pruning (non-associative f64 tie-reordering; also mutation-tested its own suite and corrected a
  wrong proof comment), filtered_vector_search (two-hop gate derived from γ fought itself; integer-
  histogram misrouting the planner), click_model (PBM scale non-identifiability; textbook-DR invalid on
  latent examination), index_maintenance (bridge-alone degrades navigability — needs Vamana refine),
  bandit_ranker (`max_by_key` tie-break bug), watermarking (degenerate γ p-value = 0.5 not 1.0).

## v0.22.0 — Classical IR Reimagined, Multi-Agent Reasoning, Domain-Specialized Retrieval & Retrieval Governance ✅

**Released**: 2026-07-02 | **Tests**: 10,958 | **Warnings**: 0

Twelve cutting-edge RAG modules, four themes, zero new deps, 185 module dirs. +835 tests from v0.21.0.

**Theme 1 — Classical IR Reimagined**
- [x] **Learning-to-Rank** (`learning-to-rank`): trained pairwise-logistic RankNet-lite over numeric
  retrieval signals (BM25/recency/embedding-sim/popularity/length), convergence proven via closed-form
  gradient checks. `LtrEngine`. 70 tests.
- [x] **RP-Tree Index** (`rp-tree-index`): Annoy-style random-projection tree forest, priority-queue
  multi-tree search + exact re-rank; recall improves with tree count (measured). `RpTreeIndex`. 59 tests.
- [x] **BM25F** (`bm25f-retrieval`): field-weighted multi-field BM25, per-field weight/`b`, combined
  pseudo-frequency before saturation. `Bm25fIndex`. 61 tests.

**Theme 2 — Multi-Agent & Principled Reasoning**
- [x] **Multi-Agent Debate** (`multi-agent-debate`): adversarial opposing personas rebut across rounds
  over the full prior transcript, judged by a separate judge. `DebateEngine`. 93 tests.
- [x] **Constitutional Critique** (`constitutional-critique`): fixed named-principle critique-and-revise
  loop, later principles see earlier revisions (proven sequential). `ConstitutionalEngine`. 86 tests.
- [x] **Tool Retrieval** (`tool-retrieval`): semantic tool/API-spec retrieval + type-aware argument
  grounding, scoped around (not reimplementing) `agentic::Tool`. `ToolRetrievalEngine`. 71 tests.

**Theme 3 — Domain-Specialized Retrieval**
- [x] **Code Retrieval** (`code-retrieval`): structural brace/keyword/indentation tokenizer blended with
  text similarity; structural signal proven to flip ranking adversarially. `CodeRetrievalEngine`. 72 tests.
- [x] **Extractive QA** (`extractive-qa`): SQuAD-style question-type-aware span-boundary extraction, no
  generation. `ExtractiveQaEngine`. 88 tests.
- [x] **Hard Negative Mining** (`hard-negative-mining`): ANCE-style async re-mining under a
  version-parameterized embedding, staleness/overlap metric measures drift. `HardNegativeMiner`. 51 tests.

**Theme 4 — Retrieval Governance**
- [x] **Active Learning Retrieval** (`active-learning-retrieval`): margin + entropy uncertainty sampling
  over an unlabeled pool, verified against hand-computed ground truth. `ActiveLearningSelector`. 60 tests.
- [x] **Belief Revision** (`belief-revision`): genuine Bayesian log-odds posterior updates, numerically
  stable under 150+ sequential updates, proven vs hand-computed Bayes rule. `BeliefRevisionEngine`. 70 tests.
- [x] **Shard Selection** (`shard-selection`): CORI-style pre-query resource selection from shard
  statistics; explicitly does not merge/calibrate (that's `ensemble_retriever`/`rank_fusion`/
  `collections`). `ShardSelector`. 63 tests.

**Integration & Release**
- [x] 12 features + 4 umbrellas (`classical-ir`, `multi-agent-reasoning`, `domain-specialized-retrieval`,
  `retrieval-governance`); 12 modules + 126 prelude re-exports, **1 alias needed**
  (`extractive_qa::QuestionType` → `ExtractiveQuestionType`, collides with the pre-existing
  `synthetic_eval::QuestionType`) — found via the same systematic name-grep check used every cycle.
- [x] `cargo fmt` + `cargo build/clippy/nextest/doc --all-features --all-targets -D warnings` → 0 errors,
  0 warnings, 10,958 tests + 168 doctests green.
- [x] Bumped 0.21.0 → 0.22.0; updated CHANGELOG.md, TODO.md.
- [x] Process note: pre-validated via 3 parallel recon passes over ~50 existing modules before writing
  any code — dropped 3 confirmed duplicates (`rocchio_feedback`≈`relevance_feedback`,
  `sentence_window_retrieval`≈`parent_document`'s reserved `WindowConfig`, `multimodal_rag`≈
  `layer1_echo`'s real CLIP provider) and re-scoped 1 (`federated_retrieval`→`shard_selection`, narrowed
  to pre-query selection since post-query merge/calibration already existed 3x over). One subagent
  (`rp_tree_index`, first attempt) failed immediately by trying to spawn its own sub-subagents instead
  of doing the work (0 tool uses, 6s) — re-dispatched with an explicit "do this yourself" instruction
  and it completed cleanly on retry. Orchestrator ran on Sonnet 5, not Opus per `/ucont` Rule 0 (no user
  response to the model-switch prompt within the wait window; proceeded per "best judgment," matching
  how v0.19.0–v0.21.0 already ran this session) — compensated with the same rigorous recon process.

## v0.21.0 — Graph-Knowledge RAG, Late-Interaction Retrieval, Tabular/Structured Knowledge & Retrieval Control ✅

**Released**: 2026-07-02 | **Tests**: 10,123 | **Warnings**: 0

Twelve cutting-edge RAG modules, four themes, zero new deps, 173 module dirs. +817 tests from v0.20.0.

**Theme 1 — Graph-Structured Knowledge RAG**
- [x] **G-Retriever** (`g-retriever`): subgraph retrieval as Prize-Collecting Steiner Tree — genuine
  Goemans–Williamson primal-dual + Johnson–Minkoff–Phillips strong pruning. `GRetrieverEngine`. 57 tests.
- [x] **Think-on-Graph** (`think-on-graph`): bounded-width beam search over KG paths with relevance
  pruning + sufficiency early-stop. `TogEngine`. 59 tests.
- [x] **LightRAG** (`lightrag`): dual-keyword-type (low entity + high thematic) Local/Global/Hybrid
  retrieval over an incrementally-deduped graph+vector index, not community-based. `LightRagEngine`. 67 tests.

**Theme 2 — Fine-Grained & Late-Interaction Retrieval**
- [x] **COIL** (`coil-retrieval`): contextualized inverted lists — exact-lexical-gated per-occurrence
  contextual dot-products + optional CLS term. `CoilRetriever`. 59 tests.
- [x] **MUVERA** (`muvera`): multi-vector → single-vector MIPS via Fixed Dimensional Encodings (SimHash
  partition + JL projection + repetitions) approximating Chamfer. `MuveraEncoder`/`MuveraIndex`. 68 tests.
- [x] **Instruction-Embed** (`instruction-embed`): INSTRUCTOR/TART instruction-conditioned embeddings
  (`base ⊙ gate + shift`) so one corpus ranks differently per task. `InstructionEmbedder`. 99 tests.

**Theme 3 — Tabular & Structured Knowledge**
- [x] **TableRAG** (`table-rag`): two-stage schema-retrieval + cell-retrieval over large tables with
  distinct-value capping and sub-table assembly. `TableRagEngine`. 70 tests.
- [x] **StructRAG** (`structrag`): router infers the optimal structure (table/graph/tree/catalogue/
  algorithm), restructures passages into it, reasons over it. `StructRagRouter`/`StructRagEngine`. 85 tests.
- [x] **Chain-of-Table** (`chain-of-table`): symbolic table-operation chain (add_column/select/group_by/
  sort_by/aggregate) evolving a relational table state. `ChainOfTableEngine`. 82 tests.

**Theme 4 — Reranking, Adaptive Control & Evaluation**
- [x] **Setwise Rerank** (`setwise-rerank`): k-way set comparison as the sort primitive inside
  heapsort/bubblesort (k=2 matches pairwise bound). `SetwiseReranker`. 52 tests.
- [x] **SKR** (`skr`): memory-based kNN retrieve-or-not gate over a labeled self-knowledge pool +
  maintenance API. `SkrGate`/`SelfKnowledgePool`. 55 tests.
- [x] **CRUD-RAG** (`crud-rag`): eval harness partitioned by Create/Read/Update/Delete operations with
  genuine ROUGE-L/BLEU/EM-F1/coverage metrics. `CrudRagHarness`. 64 tests.

**Integration & Release**
- [x] 12 features + 4 umbrellas (`graph-knowledge-rag`, `late-interaction`, `structured-knowledge`,
  `retrieval-control`); 12 modules + 142 prelude re-exports, **zero aliasing needed** (all names
  module-prefixed and collision-free — verified via systematic name-grep).
- [x] `cargo fmt` + `cargo build/clippy/nextest/doc --all-features --all-targets -D warnings` → 0 errors,
  0 warnings, 10,123 tests + 153 doctests green.
- [x] Bumped 0.20.0 → 0.21.0; updated CHANGELOG.md, TODO.md.
- [x] Process note: shared-branch git-reset hazard did NOT recur this cycle — Phase-0 wiring + all 12
  untracked module dirs survived intact (verified before Phase F). Subagent briefs forbidding git
  commands worked (2 agents ran only read-only `git log`/`status`, non-mutating).

## v0.20.0 — Learned Quantization, Reasoning Architectures, Context Efficiency & Robust Eval ✅

**Released**: 2026-07-02 | **Tests**: 9,306 | **Warnings**: 0

Twelve cutting-edge RAG modules, four themes, zero new deps, 161 module dirs.

**Theme 1 — Learned Vector Compression**
- [x] **Anisotropic VQ** (`anisotropic-vq`): ScaNN-style parallel/orthogonal-weighted quantization
  loss — weighted-least-squares centroid update, not a plain mean. `AnisotropicQuantizer`.
- [x] **ITQ** (`itq-hashing`): Iterative Quantization — PCA + alternating minimization learning an
  orthogonal rotation via orthogonal Procrustes (pure-Rust Jacobi SVD). `ItqHasher`.
- [x] **Residual VQ** (`residual-vq`): Sequential multi-stage codebook cascade over residuals, with
  additive lookup-table distance estimation and optional beam search. `ResidualQuantizer`.

**Theme 2 — Decoupled & Persistent Reasoning Architectures**
- [x] **ReWOO** (`rewoo`): Single upfront Planner pass with placeholder evidence variables, one
  retrieval call per step, zero interleaving. `RewooPlanner`/`RewooWorker`/`RewooSolver`.
- [x] **Search-in-the-Chain** (`searchain`): Upfront global reasoning chain + Interactive
  Reasoning-Verification with transitive multi-level backtracking. `SearChainEngine`.
- [x] **Buffer of Thoughts** (`buffer-of-thoughts`): Persistent, growing meta-buffer of distilled
  thought templates, retrieved by problem-structure similarity. `ThoughtBuffer`/`BotEngine`.

**Theme 3 — Prompt & Context Efficiency**
- [x] **LLMLingua** (`llmlingua`): Self-fit interpolated n-gram surrogate LM drives coarse-to-fine
  perplexity-based pruning with proportional water-filling budget allocation. `PerplexityCompressor`.
- [x] **Memory Paging** (`memory-paging`): MemGPT-style OS-inspired paging between bounded main
  context and unbounded archival storage, self-directed eviction. `ContextPager`.
- [x] **UPRISE Retrieval** (`uprise-retrieval`): Cross-task prompt-exemplar retrieval reranked by
  EMA-updated outcome quality. `UpriseRetriever`.

**Theme 4 — Statistical Calibration & Privacy Robustness**
- [x] **Conformal RAG** (`conformal-rag`): Split conformal prediction — distribution-free
  finite-sample quantile threshold with a proven marginal coverage guarantee. `ConformalCalibrator`.
- [x] **ChainPoll** (`chainpoll`): Deterministic multi-formulation chain-of-thought hallucination
  polling (not stochastic sampling) with majority-vote calibration. `ChainPollScorer`.
- [x] **Membership Inference** (`membership-inference`): Canary-based privacy audit with a genuine
  rank-based (Mann-Whitney U) AUC leakage score + defense. `CanaryAuditor`.

**Integration & Release**
- [x] 12 features + 4 umbrellas (`advanced-quantization`, `reasoning-architectures`,
  `context-efficiency`, `robust-eval`); 12 modules + prelude re-exports (2 aliased:
  `searchain::Generator`→`SearchainGenerator`, `searchain::Retriever`→`SearchainRetriever` —
  `dragin` already exports bare `Retriever`; `Generator` aliased alongside it for symmetry).
- [x] `cargo fmt` + `cargo build/clippy/nextest/doc --all-features --all-targets` → 0 errors, 0
  warnings, 9,306 tests green (+537 from v0.19.0).
- [x] Bumped 0.19.0 → 0.20.0; updated CHANGELOG.md, TODO.md.
- [x] Process note: multiple subagents independently observed a concurrent process repeatedly
  running `git reset` back to HEAD during this session (visible in `git reflog`), transiently
  wiping tracked-file changes (Cargo.toml/lib.rs wiring) but never touching untracked module
  directories — no code was lost, all 12 modules verified intact and green after the fact.

## v0.19.0 — Scalable-Indexing, Query-Transformation, Reasoning-Compression & Trust-Eval ✅

**Released**: 2026-07-02 | **Tests**: 8,769 | **Warnings**: 0

Twelve cutting-edge RAG modules, four themes, zero new deps, 149 module dirs.

**Theme 1 — Scalable Indexing & Quantization**
- [x] **DiskANN / Vamana** (`disk-ann`): Single-layer graph ANN — medoid entry, `RobustPrune(α)`
  occlusion pruning, two-pass build. `DiskAnnIndex`, `VamanaGraph`.
- [x] **SPANN** (`spann`): Balanced-clustering memory-disk hybrid ANN — boundary-closure replication +
  RNG-rule pruning. `SpannIndex`.
- [x] **RaBitQ** (`rabitq`): Randomized-rotation 1-bit quantization with an unbiased distance estimator
  + error bound. `RaBitQuantizer`, `RaBitQIndex`.

**Theme 2 — Query Transformation & Clarification**
- [x] **RQ-RAG** (`rq-rag`): Learn-to-refine query router — Rewrite/Decompose/Disambiguate/Respond.
  `QueryRefinementEngine`.
- [x] **Tree of Clarifications** (`tree-of-clarifications`): Ambiguous-question disambiguation tree with
  recursive pruning + aggregated answer. `ToCEngine`.
- [x] **Query2Doc** (`query2doc`): Pseudo-document expansion — repetition-weighted query + generated
  doc. `Query2DocExpander`.

**Theme 3 — Retrieval-Augmented Reasoning & Compression**
- [x] **RAT** (`retrieval-augmented-thoughts`): Per-thought-step revision conditioned on step-targeted
  retrieval. `RatEngine`.
- [x] **RECOMP** (`recomp`): Dual extractive/abstractive-lite compressors + selective-augmentation gate.
  `RecompPipeline`.
- [x] **FILCO** (`filco`): Sentence-granularity content filtering — STRINC/lexical/CXMI-lite measures.
  `FilcoFilter`.

**Theme 4 — Hallucination Detection & Trustworthy Eval**
- [x] **SelfCheckGPT** (`selfcheckgpt`): Zero-resource sampling-consistency hallucination scoring —
  n-gram/NLI-lite/QA-lite variants. `SelfCheckScorer`.
- [x] **eRAG** (`erag`): Per-document downstream-task retriever evaluation + Kendall/Spearman
  correlation. `ERagEvaluator`.
- [x] **EigenScore** (`eigenscore`): Differential-entropy hallucination detection via a pure-Rust Jacobi
  eigensolver. `EigenScoreDetector`.

**Integration & Release**
- [x] 12 features + 4 umbrellas (`scalable-indexing`, `query-transformation`, `reasoning-compression`,
  `trust-eval`); 12 modules + prelude re-exports (2 aliased: `QueryRefiner`→`RqRagQueryRefiner`,
  `MockRefiner`→`RqRagMockRefiner` — `corrective_rag`/`self_refine` already export those names).
- [x] Fixed 3 unrelated pre-existing doctest regressions found during verification (edition-2024
  `gen` reserved keyword in `retrieval_loop::generator`, a stale borrow in `advanced_retrieval::mmr`,
  and 3 stale `chunking::strategies` doctests missing `.with_min_chunk_size(1)`).
- [x] `cargo fmt` + `cargo build/clippy/nextest/doc --all-features --all-targets` → 0 errors, 0
  warnings, 8,769 tests green (+1,015 from v0.18.0).
- [x] Bumped 0.18.0 → 0.19.0; updated CHANGELOG.md, TODO.md.

---

## v0.18.0 — Vector-Indexing, Advanced-Prompting, Rerank-Quality & Eval-Safety ✅

**Released**: 2026-06-14 | **Tests**: 7,754 | **Warnings**: 0

Twelve cutting-edge RAG modules, four themes, zero new deps, 137 module dirs.

**Theme 1 — ANN & Vector Indexing**
- [x] **HNSW Index** (`hnsw`): Hierarchical Navigable Small World ANN — multi-layer graph, greedy beam search with `ef_construction`/`ef_search`. `HnswIndex`.
- [x] **LSH Index** (`lsh`): Locality-Sensitive Hashing — random-hyperplane (cosine) + banded MinHash (Jaccard). `LshIndex`, `MinHashIndex`.
- [x] **Scalar Quantization** (`scalar-quantization`): int8/binary quantization — calibrate, encode, decode, asymmetric dot, Hamming distance. `ScalarQuantizer`.

**Theme 2 — Advanced Prompting & Reasoning**
- [x] **Step-Back Prompting** (`step-back`): Abstract query → retrieve → synthesize. `StepBackEngine`.
- [x] **Least-to-Most** (`least-to-most`): Decompose → solve sequentially with prior answers. `LtmEngine`.
- [x] **Self-Discover** (`self-discover`): SELECT/ADAPT/IMPLEMENT atomic reasoning modules. `SelfDiscoverEngine`.

**Theme 3 — Reranking & Retrieval Quality**
- [x] **Pairwise Rerank** (`pairwise-rerank`): Tournament-style pairwise comparison with win accumulation. `PairwiseReranker`.
- [x] **Multi-Query** (`multi-query`): Generate N query variants + RRF fusion. `MultiQueryGenerator`.
- [x] **Citation Verification** (`citation-verification`): Verify claim grounding via token overlap + n-gram. `CitationVerifier`.

**Theme 4 — Evaluation & Safety**
- [x] **Faithfulness Eval** (`faithfulness-eval`): RAGAS claim-level faithfulness via NLI-lite. `FaithfulnessEvaluator`.
- [x] **Prompt Injection Defense** (`prompt-injection-defense`): 40+ pattern detection, Quarantine/Sanitize/Flag strategies. `PromptInjectionDetector`.
- [x] **Query Difficulty** (`query-difficulty`): Predict Easy/Medium/Hard/Ambiguous band. `DifficultyPredictor`.

## v0.17.0 — ANN-Indexing, Generation-Refinement, Robustness-Privacy & Advanced-Eval ✅

**Released**: 2026-06-14 | **Tests**: 6,945 | **Warnings**: 0

Twelve more cutting-edge RAG technique modules across four themes (pure-Rust heuristic, zero new deps).
Full details in `CHANGELOG.md`. Per-module test counts: PQ 71, IVF 54, PLAID 65, self-refine 70,
chain-of-density 65, analogical 56, poisoning 66, anonymization 54, abstention 60, ragchecker 54,
retrieval-diversity 60, ares 52.

**Theme 1 — ANN Indexing & Late Interaction**
- [x] **Product Quantization** (`product-quantization`): PQ — split vectors into subspaces, quantize each
  to a learned codebook, approximate distance via lookup tables (ADC). `ProductQuantizer`, `PqIndex`.
  Distinct from scalar `quantization` (INT8/INT4).
- [x] **IVF Index** (`ivf-index`): inverted-file ANN — coarse quantizer (centroids) + inverted lists;
  probe `nprobe` nearest cells. `IvfIndex.build()/search()`. Distinct from HNSW (`ann`).
- [x] **PLAID** (`plaid`): ColBERTv2/PLAID centroid-pruned MaxSim late interaction — token-level scoring
  with centroid-based candidate generation. `PlaidRetriever`. Distinct from basic `multi_vector`.

**Theme 2 — Generation Refinement**
- [x] **Self-Refine** (`self-refine`): Madaan 2023 — iterative self-feedback → refine loop (no external
  memory/retrieval, unlike Reflexion). `SelfRefineEngine.run`.
- [x] **Chain-of-Density** (`chain-of-density`): Adams 2023 — iteratively densify a fixed-length summary
  by adding salient missing entities. `ChainOfDensityEngine`, `DensityStep`.
- [x] **Analogical Prompting** (`analogical`): Yasunaga 2023 — self-generate relevant exemplars/knowledge
  before solving. `AnalogicalEngine.run`.

**Theme 3 — Robustness & Privacy**
- [x] **Poisoning Defense** (`poisoning-defense`): detect adversarial/poisoned passages (keyword-stuffing,
  anomalous repetition, consensus contradiction). `PoisoningDetector.scan()`.
- [x] **Anonymization** (`anonymization`): PII pseudonymization with a consistent reversible mapping
  (entity → placeholder, restorable). `Anonymizer.anonymize()/deanonymize()`. Distinct from guardrails PII.
- [x] **Abstention** (`abstention`): selective prediction — decide when to REFUSE based on retrieval
  support + confidence with a risk-coverage tradeoff. `AbstentionPolicy.decide()`.

**Theme 4 — Advanced Evaluation**
- [x] **RAGChecker** (`ragchecker`): Ru 2024 — claim-level diagnostics: retriever (claim recall/precision)
  + generator (faithfulness, hallucination, noise sensitivity) over claim entailment. `RagChecker`.
- [x] **Retrieval Diversity** (`retrieval-diversity`): diversity/coverage metrics — α-nDCG, subtopic
  recall (S-recall), intra-list diversity. `DiversityMetrics`.
- [x] **ARES Eval** (`ares-eval`): Saad-Falcon 2023 — automated eval with PPI (prediction-powered
  inference) confidence intervals over judge predictions. `AresEvaluator`, `PpiInterval`.

**Integration & Release**
- [x] 12 features + 4 umbrellas (`ann-indexing`, `generation-refinement`, `robustness-privacy`,
  `advanced-eval`); 12 modules + prelude re-exports (1 aliased: `PiiKind`→`AnonPiiKind`).
- [x] `cargo fmt` + `cargo build/clippy/nextest/doc --all-features --all-targets` → 0 errors, 0 warnings, 6,945 tests green.
- [x] Bumped 0.16.0 → 0.17.0; updated CHANGELOG.md, TODO.md.

---

## v0.16.0 — Graph-Generative, Retrieval-Composition, Time-Lang-Personal & Grounding-Verification ✅

**Released**: 2026-06-14 | **Tests**: 6,218 | **Warnings**: 0

Twelve more cutting-edge RAG technique modules across four themes (pure-Rust heuristic, zero new deps).
Full details in `CHANGELOG.md`. Per-module test counts: drift 54, entity-link 65, gen-retrieval 68,
auto-merge 61, ensemble 59, gen-read 51, fresh 69, cross-lingual 74, personalized 55, quote 54,
claim-decomp 56, FiD 56.

**Theme 1 — Graph & Generative Retrieval**
- [x] **DRIFT Search** (`drift-search`, dep `graph-summarization`): GraphRAG DRIFT (Microsoft 2024) —
  global community search + local entity search + iterative follow-up drill-down. `DriftSearchEngine`.
- [x] **Entity Linking** (`entity-linking`): mention → canonical KB entity disambiguation via alias map
  + context scoring. `EntityLinker.link()`, `EntityCatalog`, alias resolution.
- [x] **Generative Retrieval** (`generative-retrieval`): DSI-style — hierarchical semantic doc-id
  assignment + constrained traversal "generation". `SemanticDocId`, `GenerativeRetriever`.

**Theme 2 — Retrieval Composition**
- [x] **Auto-Merging** (`auto-merging`, dep `chunking`): LlamaIndex auto-merging — collapse retrieved
  child chunks into their parent when enough siblings are hit. `AutoMergingRetriever`.
- [x] **Ensemble Retriever** (`ensemble-retriever`): weighted ensemble of pluggable retriever traits
  with fusion. `EnsembleRetriever`, `RetrieverWeight`.
- [x] **GenRead** (`gen-read`): generate-then-read (Yu 2023) — generate diverse contextual docs,
  cluster, use as context. `GenReadEngine`, clustering.

**Theme 3 — Time, Language & Personalization**
- [x] **Fresh Retrieval** (`fresh-retrieval`): FreshLLM-style time-sensitivity detection + freshness
  scoring + staleness flagging. `FreshnessAnalyzer`.
- [x] **Cross-Lingual** (`cross-lingual`): cross-lingual matching via bilingual-lexicon normalization +
  shared pseudo-embedding space. `CrossLingualRetriever`, `BilingualLexicon`.
- [x] **Personalized RAG** (`personalized-rag`): user-profile-aware reranking blending relevance with
  topic-interest profile + history. `UserProfile`, `PersonalizedReranker`.

**Theme 4 — Grounding & Fine-grained Verification**
- [x] **Quote Grounding** (`quote-grounding`): extract minimal verbatim supporting quotes per answer
  claim (span-level grounding). `QuoteGrounder.ground()`.
- [x] **Claim Decomposition** (`claim-decomposition`): FActScore-style atomic-fact decomposition of a
  generated answer for fine-grained verification. `AtomicClaimExtractor`.
- [x] **Fusion-in-Decoder** (`fusion-in-decoder`): FiD-style per-passage evidence extraction → weighted
  cross-passage fusion with attribution. `FusionInDecoder.fuse()`.

**Integration & Release**
- [x] 12 features + 4 umbrellas (`graph-generative`, `retrieval-composition`, `time-lang-personal`,
  `grounding-verification`); 12 modules + prelude re-exports (2 aliased: `ChunkHierarchy`→`AutoMergeHierarchy`,
  `ClaimExtractor`→`AtomicClaimExtractorTrait`).
- [x] `cargo fmt` + `cargo build/clippy/nextest/doc --all-features --all-targets` → 0 errors, 0 warnings, 6,218 tests green.
- [x] Bumped 0.15.0 → 0.16.0; updated CHANGELOG.md, TODO.md.

---

## v0.15.0 — Next-Gen-Retrieval, Adaptive-Generation, Knowledge-Context & Eval-Benchmarking ✅

**Released**: 2026-06-14 | **Tests**: 5,496 | **Warnings**: 0

Twelve more cutting-edge RAG technique modules (mostly 2023-2024 papers) across four themes
(pure-Rust heuristic, zero new deps). Full details in `CHANGELOG.md`. Per-module test counts:
hippo 55, long-rag 53, dragin 65, astute 70, self-route 65, spec-draft 66, memorag 60,
ctx-pruning 61, knowledge-conflict 61, rgb 54, nugget 71, ab-eval 62.

**Theme 1 — Next-Gen Retrieval Architectures**
- [x] **HippoRAG** (`hipporag`): Gutiérrez 2024 — Personalized PageRank over an entity graph for
  single-step multi-hop retrieval. `HippoRagIndex.build()/search()`, PPR seeded from query entities.
- [x] **LongRAG** (`long-rag`): Jiang 2024 — long retrieval units (group related chunks into fewer,
  longer units) to cut unit count + boost recall. `LongUnitGrouper`, `LongRagRetriever`.
- [x] **DRAGIN** (`dragin`): Su 2024 — Dynamic RAG on real-time Information Need: decide WHEN to
  retrieve (token uncertainty / RIND) + WHAT (attention-salience query formulation / QFS).

**Theme 2 — Adaptive Generation Strategies**
- [x] **Astute RAG** (`astute-rag`): Wang 2024 — consolidate internal (parametric) vs external
  (retrieved) knowledge, detect & resolve conflicts, prefer reliable source. `AstuteConsolidator`.
- [x] **Self-Route** (`self-route`): Li 2024 — route between RAG and long-context based on
  answerability; fall back to full-context when retrieval is insufficient. `SelfRouter.route()`.
- [x] **Speculative Drafting** (`speculative-drafting`): Wang 2024 Speculative RAG — cluster docs,
  draft multiple answers in parallel from diverse subsets, verify/score drafts. `SpeculativeDrafter`.

**Theme 3 — Knowledge & Context Management**
- [x] **MemoRAG** (`memorag`): Qian 2024 — global memory gist of the corpus generates retrieval clues,
  then retrieves evidence. `MemoryGist`, `ClueGenerator`, `MemoRagEngine`.
- [x] **Context Pruning** (`context-pruning`): LLMLingua-style token-level compression to a target
  budget via token importance/perplexity. `TokenPruner.prune()`, compression ratio. (Distinct from
  sentence-extractive `context_compression`.)
- [x] **Knowledge Conflict** (`knowledge-conflict`): inter-passage contradiction detection + resolution
  (recency/authority/majority policies). `ConflictDetector`, `ConflictResolver`.

**Theme 4 — Evaluation & Benchmarking**
- [x] **RGB Eval** (`rgb-eval`): Chen 2023 — 4-ability RAG benchmark: noise robustness, negative
  rejection, information integration, counterfactual robustness. `RgbEvaluator`.
- [x] **Nugget Eval** (`nugget-eval`): nugget-based answer scoring — decompose gold answer into
  information nuggets, score by coverage (vital/okay weighting). `NuggetScorer`.
- [x] **A/B Eval** (`ab-eval`): paired A/B pipeline comparison with bootstrap significance + win-rate.
  `AbEvaluator.compare()`, `AbResult`.

**Integration & Release**
- [x] 12 features + 4 umbrellas (`next-gen-retrieval`, `adaptive-generation`, `knowledge-context`,
  `eval-benchmarking`); 12 modules + prelude re-exports (no collisions/aliases needed this round).
- [x] `cargo build/clippy/nextest/doc --all-features --all-targets` → 0 errors, 0 warnings, 5,496 tests green.
- [x] Bumped 0.14.0 → 0.15.0; updated CHANGELOG.md, TODO.md.

---

## v0.14.0 — Retrieval-Indexing, Ranking-Fusion, Structured-Reasoning & Verification-Robustness ✅

**Released**: 2026-06-14 | **Tests**: 4,753 | **Warnings**: 0

Twelve more cutting-edge RAG technique modules across four themes (pure-Rust heuristic
implementations, zero new dependencies). Full details in `CHANGELOG.md`. Per-module test counts:
sparse 56, self-query 65, summary 58, rank-fusion 55, diversity 60, credibility 75, GoT 55,
SoT 48, PoT 74, fact-check 60, noise-filter 53, answer-calibration 81.

**Theme 1 — Retrieval & Indexing**
- [x] **Sparse Retrieval** (`sparse-retrieval`): SPLADE-style learned sparse term-weighting
  (`log(1+ReLU(w))` saturation) + co-occurrence term expansion. `SparseEncoder`, `SparseIndex.search()`.
- [x] **Self-Query** (`self-query`): self-querying retriever — parse a natural-language query into a
  structured `MetadataFilter` + residual semantic query. `SelfQueryParser.parse()`, `SelfQueryRetriever`.
- [x] **Summary Index** (`summary-index`): document-summary index — index by extractive summary, retrieve
  the full parent document. `SummaryIndex.add_document()/search()`, `DocumentSummary`.

**Theme 2 — Ranking & Fusion**
- [x] **Rank Fusion** (`rank-fusion`): CombSUM/CombMNZ/Borda/ISR/weighted multi-list fusion (beyond RRF).
  `RankFusion.fuse()`, `FusionMethod`, score normalization.
- [x] **Diversity Rank** (`diversity-rank`): DPP-inspired diverse subset selection (global vs MMR greedy).
  `DiversityRanker.select()`, kernel determinant gain.
- [x] **Source Credibility** (`source-credibility`): authority scoring — citation PageRank-lite + recency +
  metadata-authority signals. `CredibilityScorer.score()`, `SourceGraph`.

**Theme 3 — Structured Reasoning**
- [x] **Graph-of-Thoughts** (`graph-of-thought`): Besta 2023 — DAG of thoughts with aggregation/refinement
  beyond ToT. `GraphOfThoughtEngine.run`, `ThoughtGraph`, aggregate/refine operations.
- [x] **Skeleton-of-Thought** (`skeleton-of-thought`): Ning 2023 — generate answer skeleton then expand
  points in parallel. `SkeletonOfThoughtEngine.run`, `SkeletonPoint`.
- [x] **Program-of-Thoughts** (`program-of-thought`): Chen 2022 — separate reasoning (program steps) from
  computation via a deterministic interpreter. `ProgramOfThoughtEngine.run`, `ProgramStep`, `Interpreter`.

**Theme 4 — Verification & Robustness**
- [x] **Fact Check** (`fact-check`): FEVER-style claim → evidence retrieval → SUPPORTS/REFUTES/NEI verdict.
  `FactChecker.verify()`, `Verdict`, evidence aggregation.
- [x] **Noise Filter** (`noise-filter`): RAAT-lite robustness — detect & downweight irrelevant/distracting
  passages before generation. `NoiseFilter.filter()`, distractor scoring.
- [x] **Answer Calibration** (`answer-calibration`): answer-level confidence calibration with verbalized +
  agreement signals, ECE/Brier/reliability-diagram metrics. `AnswerCalibrator`, `CalibrationMetrics`.

**Integration & Release**
- [x] 12 features + 4 umbrellas (`retrieval-indexing`, `ranking-fusion`, `structured-reasoning`,
  `verification-robustness`); 12 modules + prelude re-exports (3 aliased to avoid collisions:
  `SparseVector`→`LearnedSparseVector`, `FieldType`→`FilterFieldType`, `ThoughtGenerator`→`GotThoughtGenerator`).
- [x] `cargo build/clippy/nextest --all-features --all-targets` → 0 errors, 0 warnings, 4,753 tests green.
- [x] Bumped 0.13.0 → 0.14.0; updated CHANGELOG.md, TODO.md.

---

## v0.13.0 — Index-Representation, Rerank-Selection, Compositional-Reasoning & Calibration-Geometry ✅

**Released**: 2026-06-14 | **Tests**: 4,013 | **Warnings**: 0

Twelve new cutting-edge RAG technique modules across four themes (pure-Rust heuristic
implementations, zero new dependencies). Full details in `CHANGELOG.md`.

**Theme 1 — Index-Time Representation**
- [x] **Late Chunking** (`late-chunking`): Jina-2024 context-aware chunk embeddings — pool contextual
  token vectors *after* full-document encoding. `LateChunker.encode_document()`, Mean/Max pooling. 58 tests.
- [x] **Proposition Retrieval** (`proposition-retrieval`): Dense-X (Chen 2023) atomic-proposition
  decomposition + `PropositionIndex` → parent-doc retrieval. 59 tests.
- [x] **Doc2Query Expansion** (`doc2query`): index-time hypothetical-query expansion (doc2query/HyPE).
  `Doc2QueryExpander.expand()`, `ExpandedDocument`. 48 tests.

**Theme 2 — Reranking & Result-Set Selection**
- [x] **Listwise Reranking** (`listwise-rerank`): RankGPT (Sun 2023) sliding-window listwise permutation
  reranking. `ListwiseReranker.rerank()`, `ListwiseJudge`, `WindowConfig`. 55 tests.
- [x] **Autocut** (`autocut`): Weaviate relevance-gap truncation — `Jumps`/`RelativeThreshold`/`StdDev`/`Knee`.
  `AutoCutter.cut()`, `AutoCutReport`. 59 tests.
- [x] **Semantic Dedup** (`semantic-dedup`): SimHash + MinHash/LSH near-duplicate clustering & removal.
  `SemanticDeduplicator.deduplicate()`, union-find single-linkage, `KeepPolicy`. 64 tests.

**Theme 3 — Compositional Reasoning**
- [x] **Self-Ask** (`self-ask`): Press-2022 explicit follow-up decomposition. `SelfAskEngine.run<M,A>`,
  `FollowUp`, `SelfAskTrace`. 52 tests.
- [x] **Self-Consistency** (`self-consistency`): Wang-2022 sample paths → cluster answers → marginalize.
  `SelfConsistencyEngine.run<S>`, `VoteWeighting`, `AnswerCluster`. 67 tests.
- [x] **Adaptive-RAG** (`adaptive-rag`): Jeong-2024 query-complexity classifier → retrieval-depth routing.
  `ComplexityClassifier`, `AdaptiveRagRouter.route()`, `RoutingPlan`. 72 tests.

**Theme 4 — Calibration, Uncertainty & Embedding Geometry**
- [x] **Semantic Entropy** (`semantic-entropy`): Kuhn-2023 meaning-cluster entropy as uncertainty signal.
  `SemanticEntropyEstimator.estimate()`, `MeaningCluster`. 66 tests.
- [x] **Matryoshka** (`matryoshka`): Kusupati-2022 nested truncatable embeddings → two-stage coarse-to-fine
  retrieval. `MatryoshkaEmbedding.truncate()`, `MatryoshkaRetriever.search()`. 68 tests.
- [x] **Synthetic Eval** (`synthetic-eval`): RAGAS/ARES-style test-set generation — `SyntheticQa` tuples +
  hard distractors. `SyntheticEvalGenerator.generate()`, `HeuristicTemplater`. 60 tests.

**Integration & Release**
- [x] 12 features + 4 theme umbrellas (`index-representation`, `rerank-selection`, `compositional-reasoning`,
  `calibration-geometry`) in `Cargo.toml`; 12 cfg-gated modules + prelude re-exports in `lib.rs`.
- [x] `cargo build/clippy/nextest --all-features --all-targets` → 0 errors, 0 warnings, 4,013 tests green.
- [x] Bumped 0.12.0 → 0.13.0; updated CHANGELOG.md, TODO.md.

---

## v0.12.0 — Reranking-Precision, Advanced-Reasoning, Memory-State & Eval-Optimization ✅

**Released**: 2026-06-10 | **Tests**: 3,285 | **Warnings**: 0

- [x] **Cross-Encoder Reranking** (`cross-encoder`): 7 interaction features, IDF from candidate set, logistic squash, `CrossEncoderReranker.rerank()`
- [x] **Contextual Retrieval** (`contextual-retrieval`): TF-IDF extractive blurb + title/position/preceding-gist, `ContextualIndexBuilder.build()`
- [x] **Lost-in-the-Middle** (`lost-in-middle`): U-shaped reorder (Sandwich/HeadTail), `LostInMiddleReorderer.reorder()`
- [x] **Reflexion** (`reflexion`): Verbal self-reflection loop, episodic memory, `ReflexionEngine.run<E>`
- [x] **Tree-of-Thoughts** (`tree-of-thought`): BFS/DFS beam search, `ThoughtTree`, `TreeOfThoughtEngine.run<E>`
- [x] **Chain-of-Verification** (`chain-of-verification`): CoVe draft→verify→revise, Jaccard verdicts, `ChainOfVerificationEngine.run<E>`
- [x] **Long-Term Memory** (`long-term-memory`): Generative-agents memory stream, recency+importance+relevance retrieval, `MemoryRetriever`
- [x] **Memory Compression** (`memory-compression`): Hierarchical compaction, `HierarchicalMemory`, `ExtractiveTurnCompressor`
- [x] **Entity Memory** (`entity-memory`): Per-entity knowledge tracking, salience scoring, `EntityMemoryStore.observe()`
- [x] **Retrieval Evaluation** (`retrieval-eval`): nDCG/MAP/MRR/P@k/R@k/AP, `RetrievalEvaluator.evaluate_batch()`
- [x] **LLM-as-Judge** (`llm-judge`): Pointwise/pairwise/reference, `HeuristicJudge`, `LlmJudge`
- [x] **Prompt Optimization** (`prompt-optimization`): KNN/MMR/Deterministic/Hardest demo selection, `PromptOptimizer.evaluate_variants()`

---

## v0.11.0 — Multi-hop, Iterative, Trust & Composition ✅

**Released**: 2026-06-10 | **Tests**: 2,687 | **Warnings**: 0

- [x] **Multi-hop Retrieval** (`multi-hop`): Entity-chain BFS traversal, `MultiHopRetriever.run<E>`
- [x] **Fact Triples** (`fact-triples`): SVO triple extraction, `TripleExtractor`, `TripleStore`
- [x] **Knowledge Graph QA** (`knowledge-graph-qa`): Subgraph + fact synthesis, `KgqaEngine`
- [x] **Iterative RAG** (`iterative-rag`): ITER-RETGEN loop, expansion terms, `IterativeRagEngine`
- [x] **Chain-of-Note** (`chain-of-note`): Per-doc extractive notes → synthesis, `ChainOfNoteEngine`
- [x] **Answer Aggregation** (`answer-aggregation`): MajorityVote/WeightedFusion/Extractive, `AnswerAggregator`
- [x] **Hallucination Detection** (`hallucination-detection`): Claim-support scoring, `HallucinationDetector`
- [x] **Consistency Checking** (`consistency-checking`): Numerical/Temporal/Negation pairwise, `ConsistencyChecker`
- [x] **Trust Scoring** (`trust-scoring`): 4-component composite score, `TrustScorer`
- [x] **Semantic Router** (`semantic-router`): FNV-1a KNN routing, `SemanticRouter.route()`
- [x] **Query Planning** (`query-planning`): DAG plans, Kahn's topo-sort executor, `PlanExecutor.run<E>`
- [x] **Pipeline Composer** (`pipeline-composer`): `PipelineStage` trait, `ComposedPipeline`

---

## v0.10.0 — Agentic, Safety, Graph-Intelligence & Retrieval-Depth ✅

**Released**: 2026-06-10 | **Tests**: 2,455 | **Warnings**: 0

- [x] **Self-RAG** (`self-rag`): Reflection tokens, `HeuristicReflector`, `SelfRagEngine.run<E>`
- [x] **Agentic / ReAct** (`agentic`): `Tool` trait, `ToolRegistry`, `ReActAgent.run<E>`, `CalculatorTool`, `LookupTool`
- [x] **Query Decomposition** (`query-decomposition`): Parallel/LeastToMost/StepBack splits, RRF recombination
- [x] **Context Compression** (`context-compression`): `ExtractiveCompressor`, `RedundancyFilter`, token-budget packing
- [x] **Guardrails** (`guardrails`): PII (char-class FSM), injection detection, content moderation, `GuardrailEngine`
- [x] **Structured Extraction** (`structured-extraction`): Keyword-proximity typed extraction, `SchemaExtractor`
- [x] **Output Validation** (`output-validation`): Rule-based answer enforcement, `OutputValidator`, 7 rule kinds
- [x] **Graph Community** (`graph-community`): Louvain modularity detection, `LouvainDetector`, `CommunityGraph`
- [x] **Graph Summarization** (`graph-summarization`): `CommunitySummarizer`, `GlobalSearchEngine`, `LocalSearchEngine`
- [x] **RAPTOR** (`raptor`): FNV-1a pseudo-embeddings, agglomerative+KMeans clustering, `RaptorTree.collapsed_retrieval`
- [x] **Parent-Document Retrieval** (`parent-document`): `ParentChildIndex`, `ParentDocumentRetriever.run<E>`
- [x] **Temporal Re-ranking** (`temporal-retrieval`): Exponential/Linear/Gaussian/None decay, ISO 8601 parser

---

## Vision Alignment

This project implements four innovative concepts:

1. **Speculative RAG** - Use cache as "drafts" not "answers", verify with SLM
2. **Context-Aware Prefix Caching** - Efficiently manage KV Cache for "premise knowledge"
3. **On-the-fly Distillation** - Automatically generate specialized lightweight models for frequent queries
4. **Hidden States Manipulation** - Direct manipulation of transformer hidden states for verification

---

## High Priority

### Speculative RAG (Core Vision #1) - 100% Complete ✅

**Goal**: Use cache as "drafts" instead of "final answers", verify with SLM

- [x] Vector search for similar documents (Layer 1: Echo)
- [x] Draft generation pipeline (`generate_draft`)
- [x] Accept/Revise/Reject decision flow (Layer 2: Speculator)
- [x] High-speed parallel processing (`tokio::join!`, `process_batch`)
- [x] **Streaming Verification**: Stream verification results with chunks
- [x] **Confidence Calibration**: Platt scaling, isotonic, temperature scaling, histogram binning
- [x] **SLM Interface**: SmallLanguageModel trait with MockSlm implementation
- [x] **Verification Pipeline**: Multi-stage verification with built-in stages
- [x] **Speculative Decoding**: Full speculative decoding with hidden states
- [x] **Real SLM Integration**: Complete Candle-based SLM with Phi-2/Phi-3 support (`candle_slm.rs`)

### Context-Aware Prefix Caching (Core Vision #2) - 98% Complete ✅

**Goal**: Manage "understanding state of loaded documents (KV Cache)" not just "answers"

- [x] Embedding Cache with LRU (Layer 1)
- [x] **KV-Cache Types**: Core types for KV cache management
- [x] **Context Fingerprinting**: Hash-based context identification and reuse
- [x] **Prefix Cache Store**: In-memory prefix cache with eviction
- [x] **Paged Cache**: PagedAttention-inspired paging with CachePage, PageTable
- [x] **Hierarchical Cache**: L1/L2/L3 tier cache with promotion/demotion
- [x] **Cache Invalidation**: TTL, MaxAge, dependency-based invalidation policies
- [x] **Persistent Backend**: File-based persistent cache with HybridPersistentCache
- [x] **External Backend**: redb (pure-Rust ACID embedded DB) — `RedbPrefixCache` with TTL eviction, prefix-match, LRU capacity enforcement (`prefix_cache/redb_backend.rs`, feature `prefix-cache-redb`)

### Hidden States (Core Vision #4) - 95% Complete ✅

**Goal**: Direct manipulation of transformer hidden states for speculative verification

- [x] **Hidden State Types**: HiddenStateTensor, LayerHiddenState, ModelHiddenStates
- [x] **KV Cache**: ModelKVCache for efficient incremental decoding
- [x] **State Provider**: HiddenStateProvider trait with MockHiddenStateProvider
- [x] **State Caching**: HiddenStateCache with LRU eviction and prefix matching
- [x] **Reuse Strategies**: Prefix, Semantic, Hybrid, Adaptive strategies
- [x] **State Similarity**: Cosine, L2, layer-wise comparison utilities
- [x] **Speculative Decoder**: SpeculativeDecoder with draft/target model architecture
- [x] **Hidden State Speculator**: HiddenStateSpeculator for verification via state comparison
- [x] **Divergence Detection**: Identify factual inconsistencies via hidden state divergence
- [x] **Candle Integration**: Real BERT-based `CandleHiddenStateProvider` with HuggingFace Hub model loading, full tokenisation pipeline, real forward-pass hidden state extraction (`hidden_states/candle_provider.rs`, feature `hidden-states + speculator`)
- [x] **Hidden State Pooling**: `HiddenStatePooling` enum (CLS/MeanPool/MaxPool/MaskMean), `apply_hidden_state_pooling` free function, `extract_sentence_embedding` for pooled sentence-level vectors

### On-the-fly Distillation (Core Vision #3) - 100% Complete ✅

**Goal**: Automatically generate specialized lightweight models (SLM) for frequent queries

- [x] **Query Frequency Tracking**: Track query pattern frequency
- [x] **Q&A Pair Collection**: Automatic collection of Q&A pairs for distillation
- [x] **Distillation Candidate Detection**: Identify patterns ready for distillation
- [x] **LoRA Training Types**: LoraConfig, TrainingJob, LoraTrainer trait, MockLoraTrainer
- [x] **Model Registry**: ModelMetadata, ModelMetrics, find by pattern, metrics tracking
- [x] **Distillation Trigger**: TriggerCondition (frequency, count, confidence, time, combined)
- [x] **Hot-swap Models**: ModelSelector with strategies (PatternMatch, LowestLatency, etc.)
- [x] **Feature Distillation**: FitNet-style intermediate layer distillation, attention transfer
- [x] **Loss Functions**: KL divergence, MSE, cosine similarity, combined losses
- [x] **Training Metrics**: Accuracy, loss tracking, early stopping criteria
- [x] **Teacher-Student**: Teacher-student architecture with inference support
- [x] **Progressive Distillation**: Multi-stage progressive knowledge transfer
- [x] **Real LoRA Training**: Complete Candle-based LoRA training system (`candle_lora.rs`)

---

## Layer-Specific Tasks

### Layer 1: Echo (Semantic Search)

- [x] In-memory vector store
- [x] Mock embedding provider
- [x] Similarity metrics (cosine, euclidean, dot product)
- [x] Embedding cache with LRU eviction
- [x] Metadata filtering for search
- [x] Document update/upsert operations
- [x] **HNSW Index**: Approximate nearest neighbor search (`ann.rs`)
- [x] **Multi-vector Documents**: ColBERT-style late interaction (`multi_vector.rs`)
- [x] **SIMD Similarity**: Hardware-accelerated similarity computation (`simd_similarity.rs`)
- [x] **Persistent Vector Store**: redb-backed `RedbVectorStore` with full `VectorStore` trait impl, persistence across restarts, similarity search (`layer1_echo/storage/redb.rs`, feature `echo-redb`)
- [x] **Real Embedding Models**: `CandleEmbeddingProvider` (all-MiniLM-L6-v2 default) + presets for BGE-base/large/small-en-v1.5 and all-mpnet-base-v2 (`embedding/candle.rs`)

### Layer 2: Speculator (Draft Verification)

- [x] Rule-based speculator
- [x] Mock SLM speculator
- [x] Streaming verification support
- [x] Confidence calibration system
- [x] SmallLanguageModel trait and MockSlm
- [x] Multi-stage verification pipeline
- [x] **Quantization Types**: INT8, INT4, Binary quantization support (`quantization.rs`)
- [x] Complete Candle SLM integration with Phi-2/Phi-3 (production-ready)

### Layer 3: Judge (SMT Verification)

- [x] Pattern-based claim extractor
- [x] Advanced claim extractor
- [x] Mock SMT verifier
- [x] Temporal, causal, and modal claim structure types
- [x] Claim deduplication and normalization
- [x] Explanation generation for verification results
- [x] **Dependency Parsing**: Improved claim extraction with SVO extraction (`dependency_parser.rs`)
- [x] **Incremental Consistency**: Incremental consistency checking with conflict detection (`incremental.rs`)
- [x] **Real OxiZ SMT solver integration**: Complete with timeout handling, 26 tests, 9 benchmarks (`oxiz_verifier.rs`)

### Layer 4: Graph (GraphRAG)

- [x] Core types (GraphEntity, GraphRelationship, GraphPath)
- [x] EntityExtractor and RelationshipExtractor traits
- [x] Mock entity/relationship extractors
- [x] Pattern-based entity/relationship extractors
- [x] In-memory graph store
- [x] BFS traversal algorithms
- [x] Shortest path and find entities within N hops
- [x] GraphLayer with builder pattern
- [x] HybridSearchResult for combining vector + graph search
- [x] **Persistent Graph Store**: redb-backed `RedbGraphStore` with full `GraphStore` trait impl, atomic entity/relationship writes, BFS traversal, NAME\_IDX/TYPE\_IDX secondary indexes, 12 unit tests (`layer4_graph/redb_store.rs`, feature `graphrag-redb`)

### Pipeline

- [x] Unified pipeline with fast-path optimization
- [x] Pipeline builder pattern
- [x] Retry logic with exponential backoff
- [x] Parallel layer execution
- [x] Metrics and tracing
- [x] Batch query processing
- [x] **Pipeline Debugging**: Visualization and tracing tools (`pipeline_debug.rs`)
- [x] **Circuit Breaker**: Resilience pattern for external service failures (`circuit_breaker.rs`)
- [x] **Observability**: `PipelineSpanContext` with RAII `LayerSpan<'_>` guard, `SpanReport` ASCII/JSON table, `record_pipeline_event` free function — zero external tracing backend required (`observability.rs`)

---

## Medium Priority

### Performance

- [x] **SIMD Optimization**: Hardware-accelerated similarity computation with ARM NEON + AVX/SSE2 (`similarity_simd.rs`)
- [x] **Connection Pooling**: Generic connection pool for external services (`connection_pool.rs`)
- [x] **Memory Monitoring**: Memory limits and usage tracking (`memory.rs`)
- [x] **Profile and optimize hot paths**: 5.6x-9.0x speedup for cosine similarity, 8x for search workloads

### WASM

- [ ] Test and optimize WASM bundle size
- [x] Add Web Worker support for background processing (v0.5.0)
- [x] Implement IndexedDB backend for persistent storage (v0.5.0)
- [x] Add streaming response support (v0.5.0 — ReadableStream via query_stream)
- [x] Create React/Vue/Svelte component wrappers (v0.6.0 — npm/ TypeScript package)

### OpenTelemetry (v0.4.0 ✅)

- [x] **`SpanObserver` trait**: pluggable observer hook in `PipelineSpanContext`
- [x] **`MemoryObserver`**: in-memory collector for tests and REST metrics endpoint
- [x] **Pipeline instrumentation**: `PipelineBuilder::with_observers`, per-layer RAII spans
- [x] **`OtelSpanObserver`**: stdout + OTLP/gRPC exporters (feature `otel`)
- [x] **Example**: `examples/otel_tracing.rs` (stdout exporter, zero infrastructure)

### API Improvements

- [x] **Query Builder**: Fluent interface for building queries (`query_builder.rs`)
- [x] **Index Management**: Rebuild, optimize, vacuum, snapshot API (`index_management.rs`)
- [x] Async streaming API for long-running queries (`streaming.rs`)

### Testing

- [x] **Load Testing**: Utilities for concurrent operations testing (`load_testing.rs`)
- [x] **Property-based testing**: 60+ proptest tests for vectors, cache, graphs, claims, normalization
- [ ] Add integration tests with real models (network-guarded tests exist; need live model run)
- [x] **Cross-layer integration tests**: `tests/pipeline_e2e.rs`, `tests/persistence_redb.rs`, `tests/observability_e2e.rs` (v0.4.0)
- [x] **Proptest fuzz harness**: `tests/fuzz_claim_extraction.rs`, `tests/fuzz_query_normalization.rs`, `tests/fuzz_fingerprint.rs` (v0.4.0)

---

## Low Priority

### Documentation

- [x] Add API documentation with examples (v0.6.0 — docs/layers/ tutorials)
- [x] Create architecture decision records (ADRs) (v0.6.0 — docs/adr/)
- [x] Write layer-specific tutorials (v0.6.0 — docs/layers/)
- [x] Add troubleshooting guide (v0.6.0 — docs/troubleshooting.md)
- [x] Create performance tuning guide (v0.5.0 — docs/perf.md)

### Ecosystem

- [x] Create Python bindings with PyO3 (v0.5.0)
- [x] Add Node.js bindings (v0.6.0 — `src/nodejs/`, napi-rs 2.x, `package.json`, `tests/nodejs_smoke.rs`)
- [x] **REST API server** (`src/rest_server.rs`, feature `rest-server`): `AppState`, `build_router`, `build_and_serve`, 5 routes, 19 tests (v0.4.0)
- [x] Create Docker image (v0.5.0)
- [x] **OpenTelemetry integration** (`src/observability/otel.rs`, feature `otel`): OTLP + stdout exporters (v0.4.0)

### Advanced Features

- [x] **Hybrid Search**: Dense + sparse (BM25) retrieval with fusion (`hybrid_search.rs`)
- [x] **Reranking**: Cross-encoder style reranking pipeline (`reranker.rs`)
- [x] **Query Expansion**: Synonyms, stemming, PRF expansion (`query_expansion.rs`)
- [x] **Relevance Feedback**: User feedback loop with Rocchio algorithm (`relevance_feedback.rs`)
- [x] **Multi-modal embeddings** (`src/layer1_echo/embedding/clip.rs`, feature `multimodal`): CLIP text+image embeddings via Candle, `EmbeddingInput<'a>` enum, `MultiModalEmbeddingProvider` trait, `CandleClipProvider` (v0.4.0)

---

## Completed (v0.1.0)

### Core Infrastructure
- [x] Core type system (Document, Query, SearchResult, Draft, etc.)
- [x] Error handling with thiserror
- [x] Configuration management (JSON serialization)
- [x] WASM bindings structure
- [x] Comprehensive test suite (1,500 tests)
- [x] Clippy compliance (no warnings)
- [x] Rustdoc compliance (no warnings)
- [x] Criterion benchmark suite
- [x] Release preparation (LICENSE, CHANGELOG, publish script)

### Speculative RAG (99%)
- [x] Draft-based pipeline architecture
- [x] Accept/Revise/Reject decision flow
- [x] Parallel layer execution (tokio::join!)
- [x] Batch query processing (process_batch)
- [x] Streaming verification with VerificationChunk
- [x] Confidence calibration (Platt, Isotonic, Temperature, Histogram)
- [x] SmallLanguageModel trait with MockSlm
- [x] Multi-stage VerificationPipeline (Keyword, Semantic, Factual stages)

### Prefix Caching (95%)
- [x] KVCacheEntry and ContextFingerprint types
- [x] PrefixCacheStore trait with async operations
- [x] InMemoryPrefixCache with LRU eviction
- [x] Context fingerprinting with rolling hash
- [x] TTL-based cache expiration
- [x] Prefix matching for partial cache hits
- [x] CachePage and PageTable for paged cache management
- [x] HierarchicalCache with L1/L2/L3 tiers
- [x] InvalidationManager with dependency tracking
- [x] InvalidationPolicy (TTL, MaxAge, MaxStale, DependencyBased)

### Distillation (85%)
- [x] QueryPattern and QAPair types
- [x] DistillationTracker trait
- [x] QueryFrequencyTracker with pattern normalization
- [x] QAPairCollector with deduplication
- [x] CandidateDetector with priority ranking
- [x] Training example export functionality
- [x] LoraConfig, TrainingJob, LoraTrainer trait
- [x] MockLoraTrainer for testing
- [x] ModelRegistry with metrics tracking
- [x] DistillationTrigger with configurable conditions
- [x] ModelSelector with hot-swap strategies
- [x] Feature-based distillation (FitNet, attention transfer)
- [x] Distillation loss functions (KL, MSE, cosine, combined)
- [x] Training metrics and evaluation utilities
- [x] Teacher-student architecture with inference
- [x] Progressive knowledge distillation

### Layer Implementations
- [x] Layer 1: Echo (Vector Search + HNSW + Multi-vector + SIMD)
- [x] Layer 2: Speculator (Rule-based + Mock + Streaming + Calibration + Quantization)
- [x] Layer 3: Judge (Claim extraction + Mock SMT + Dependency parsing + Incremental)
- [x] Layer 4: GraphRAG (Entity/Relationship extraction)

### Hidden States (Core Feature)
- [x] HiddenStateTensor and ModelHiddenStates types
- [x] KVCache and ModelKVCache for attention caching
- [x] HiddenStateProvider trait with MockHiddenStateProvider
- [x] HiddenStateCache with LRU eviction
- [x] StateReuseStrategy trait with Prefix, Semantic, Hybrid strategies
- [x] StateSimilarity utilities (cosine, L2, layer-wise)
- [x] SpeculativeDecoder with hidden state manipulation
- [x] HiddenStateSpeculator for verification
- [x] Divergence detection via hidden state comparison

### New Modules (v0.1.0)
- [x] `query_builder.rs` - Fluent query construction API
- [x] `index_management.rs` - Index rebuild, optimize, vacuum, snapshot
- [x] `simd_similarity.rs` - SIMD-accelerated vector operations
- [x] `quantization.rs` - INT8/INT4/Binary quantization types
- [x] `hybrid_search.rs` - Dense + sparse retrieval fusion
- [x] `reranker.rs` - Multi-stage reranking pipeline
- [x] `query_expansion.rs` - Query expansion and reformulation
- [x] `relevance_feedback.rs` - User feedback with Rocchio algorithm
- [x] `connection_pool.rs` - Generic connection pooling
- [x] `memory.rs` - Memory monitoring and limits
- [x] `load_testing.rs` - Load testing utilities
- [x] `pipeline_debug.rs` - Pipeline visualization and tracing
- [x] `layer1_echo/multi_vector.rs` - ColBERT-style multi-vector support
- [x] `layer3_judge/dependency_parser.rs` - Dependency parsing for claims
- [x] `layer3_judge/incremental.rs` - Incremental consistency checking
- [x] `distillation/feature.rs` - Feature-based distillation (FitNet, attention transfer)
- [x] `distillation/losses.rs` - Distillation loss functions (KL, MSE, cosine)
- [x] `distillation/metrics.rs` - Training metrics and evaluation
- [x] `distillation/teacher_student.rs` - Teacher-student architecture
- [x] `distillation/progressive.rs` - Progressive knowledge distillation

---

## Completed (v0.1.1)

### OxiZ SMT Solver Integration
- [x] Real OxiZ SMT solver implementation (`oxiz_verifier.rs`)
- [x] Timeout handling with configurable SolverConfig
- [x] 26 comprehensive tests covering all claim types
- [x] 9 performance benchmarks for solver operations
- [x] Support for predicate, numeric, temporal, causal, modal claims
- [x] Batch verification and consistency checking
- [x] Zero unwrap() in production code

### SIMD Performance Optimization
- [x] ARM NEON intrinsics for Apple Silicon (`similarity_simd.rs`)
- [x] x86_64 AVX and SSE2 intrinsics for Intel/AMD
- [x] Platform-specific optimizations with compile-time dispatch
- [x] 5.6x-9.0x speedup for cosine similarity
- [x] 8x speedup for 5000 document search workloads
- [x] Safe abstractions around unsafe SIMD code
- [x] Comprehensive performance report at `/tmp/oxirag_performance_report.md`

### Property-Based Testing
- [x] Added proptest 1.10.0 to dev-dependencies
- [x] 60+ property-based tests across modules
- [x] Vector operations: commutativity, range checks, normalization idempotence
- [x] Cache eviction: LRU correctness, size limits, deterministic behavior
- [x] Graph traversal: shortest path properties, BFS correctness, hop limits
- [x] Claim extraction: SMT-LIB generation validation
- [x] Query normalization: idempotence, consistency, whitespace handling

### Bug Fixes
- [x] Fixed `test_search_performance_scales` timing for debug builds
- [x] Eliminated unwrap() from `src/distillation/progressive.rs`
- [x] Improved error handling across codebase

### Test Suite Expansion
- [x] Expanded from 1,500 to 1,451 tests (some consolidated)
- [x] All tests passing with zero warnings
- [x] Zero clippy warnings with `--all-features`

---

## Completed (v0.3.0)

### Persistent Graph Store (`graphrag-redb`)
- [x] `RedbGraphStore` with 6 redb tables (ENTITIES, RELATIONSHIPS, OUTGOING, INCOMING, NAME\_IDX, TYPE\_IDX)
- [x] Full `GraphStore` trait implementation with atomic write transactions
- [x] BFS traversal mirroring `InMemoryGraphStore` exactly
- [x] Exact index hit + full-scan fallback for `find_entities_by_name`
- [x] Persistence across restarts with entity/relationship count restoration
- [x] 12 unit tests including persistence round-trip test
- [x] Feature gate: `graphrag-redb = ["graphrag", "dep:redb"]`

### Observability (`observability.rs`)
- [x] `PipelineSpanContext` with UUID execution id, per-context attributes, elapsed timing
- [x] `LayerSpan<'_>` RAII guard with panic-safe `Drop` (synthesises Error record on unwind)
- [x] `SpanStatus`: Success / Error(String) / Skipped / InProgress
- [x] `LayerSpanRecord` with duration\_ms, item\_count, arbitrary attributes
- [x] `SpanReport` with `format_table()` (ASCII) and `to_json()` (serde\_json)
- [x] `record_pipeline_event` free function for lightweight event logging
- [x] Zero external dependencies (uses only `tracing` + `uuid` already in scope)
- [x] 12 unit tests covering all commit paths and panic-safe Drop
- [x] Re-exported from crate prelude in `lib.rs`

### Hidden State Pooling
- [x] `HiddenStatePooling` enum: `Cls` (default), `MeanPool`, `MaxPool`, `MaskMean`
- [x] `apply_hidden_state_pooling` pure free function (testable without model loading)
- [x] `CandleHiddenStateConfig.pooling` field + `with_pooling` builder
- [x] `extract_sentence_embedding` method on `CandleHiddenStateProvider`
- [x] 6 pooling unit tests verifying exact arithmetic for each strategy

### Expanded Property-Based Testing
- [x] 8 proptest tests in `prefix_cache/redb_backend.rs`: put/get roundtrip, TTL expiry, multi-key isolation, clear semantics, capacity enforcement
- [x] 8 proptest tests in `layer1_echo/storage/redb.rs`: document roundtrip, upsert idempotence, dimension rejection, search result ordering
- [x] 7 tests in `embedding/candle.rs`: preset field validation, URL format, max\_length ranges
- [x] 9 tests in `hidden_states/candle_provider.rs`: config builder, sequence-length variants, pooling variant existence

---

## Notes

### Dependencies to Watch
- `candle`: Monitor for WASM support improvements
- `oxiz`: Check for new SMT theories and features
- `tokenizers`: Watch for WASM compatibility

### Breaking Changes Planned
- v0.2: Refactor `SearchResult` to include more metadata
- v0.2: Change `PipelineOutput` to use cow strings for efficiency
- v0.3: Unified async trait without `async-trait` crate (when stable)

### Vision Milestone Targets

| Milestone | Speculative RAG | Prefix Caching | Distillation | Hidden States |
|-----------|-----------------|----------------|--------------|---------------|
| v0.1.0    | **99%**         | **95%**        | **85%**      | **90%**       |
| v0.1.1    | 100%            | 95%            | 100%         | 90%           |
| **v0.2.0**| **100%** ✅     | **98%** ✅     | **100%** ✅  | **95%** ✅    |
| **v0.3.0**| **100%** ✅     | **99%** ✅     | **100%** ✅  | **98%** ✅    |
| **v0.4.0**| **100%** ✅     | **99%** ✅     | **100%** ✅  | **98%** ✅    |
| v1.0.0    | 100%            | 100%           | 100%         | 100%          |

### Codebase Statistics (v0.9.0)
- **Source Files**: ~185 Rust files (+20 for prompt_templates/query_router/corrective_rag/attribution sub-files)
- **Total Lines**: ~110,000+ (Rust code)
- **Tests**: 2,219 (all passing; +190 from v0.8.0)
- **Clippy Warnings**: 0
- **Rustdoc Warnings**: 0
- **New features**: `prompt-templates`, `query-routing`, `corrective-rag`, `attribution`, `adaptive-control-plane`
- **New Modules**:
  - `src/prompt_templates/` — versioned template registry, 2-phase tokenizer + recursive-descent engine, `{{#if}}`/`{{#unless}}`/`{{else}}`, 4 built-in RAG templates
  - `src/query_router/` — 10-intent classifier (heuristic signal tables), 6-strategy router, RouterConfig with full routing/fallback/top-k tables
  - `src/corrective_rag/` — CRAG grading (Correct/Ambiguous/Incorrect), KnowledgeRefiner strip decompose/recompose, QueryRefiner rewrite, MMR with lexical pseudo-embeddings
  - `src/attribution/` — token-Jaccard SentenceAligner, CitationFormatter (Numeric/Footnote/Author), FaithfulnessChecker, Attributor with stable dedup

### Codebase Statistics (v0.8.0)
- **Source Files**: ~165 Rust files (+25 for conversation/flare/collections/document_pipeline sub-files)
- **Total Lines**: ~100,000+ (Rust code)
- **Tests**: 2,029 (all passing; +188 from v0.7.0)
- **Clippy Warnings**: 0
- **Rustdoc Warnings**: 0
- **New features**: `conversational`, `flare`, `collections`, `document-pipeline`
- **New Modules**:
  - `src/conversation/` — multi-turn ConversationHistory, 4 buffer strategies, FollowUpDetector, QueryReformulator, InMemorySessionManager, ConversationalPipeline
  - `src/retrieval_loop/` — FLARE iterative retrieval loop, ConfidenceEstimator, FlareEngine, mock generator/retriever
  - `src/collections/` — namespaced multi-tenant CollectionIndex with RRF cross-collection fusion
  - `src/document_pipeline/` — IndexingPipeline (auto-chunk + dedup + provenance), RetrievalPipeline (semantic cache + MMR + provenance enrichment), DocumentPipelineBuilder

### Codebase Statistics (v0.7.0)
- **Source Files**: ~140 Rust files (+20 for chunking/evaluation/semantic_cache/advanced_retrieval sub-files)
- **Total Lines**: ~90,000+ (Rust code)
- **Tests**: 1,841 (all passing; +160 from v0.6.0)
- **Clippy Warnings**: 0
- **Rustdoc Warnings**: 0
- **New features**: `chunking`, `rag-eval`, `semantic-cache`, `advanced-retrieval`
- **New Modules**:
  - `src/chunking/` — FixedSize, Sentence, Recursive, Markdown strategies
  - `src/evaluation/` — RAGAS-style Answer Relevance, Faithfulness, Context Precision/Recall
  - `src/semantic_cache/` — cosine-similarity memoization, LRU eviction, TTL, CacheStats
  - `src/advanced_retrieval/` — RAG-Fusion (RRF), HyDE, MMR

### Codebase Statistics (v0.6.0)
- **Source Files**: ~120 Rust files (+4 nodejs, +4 streaming sub-files, +4 query_expansion sub-files, +5 circuit_breaker sub-files, +6 connection_pool sub-files, +1 nodejs_smoke test)
- **Total Lines**: ~81,000+ (Rust code)
- **Tests**: 1,681 (all passing; +11 nodejs smoke, 8 skipped WASM/network)
- **Clippy Warnings**: 0
- **Rustdoc Warnings**: 0
- **New features**: `nodejs`
- **New Ecosystem**:
  - `src/nodejs/` (napi-rs 2.x bindings: `NapiPipeline`, `NapiPipelineBuilder`, `NapiDocument`, `NapiQuery`, `NapiSearchResult`)
  - `build.rs` + `napi_stub.c` + `libnapi_stub.so` (napi linker bridge)
  - `package.json` (@cool-japan/oxirag npm package)
  - `npm/` TypeScript WASM wrapper (`@cool-japan/oxirag-wasm`)
  - `docs/adr/` (5 Architecture Decision Records)
  - `docs/layers/` (4 per-layer tutorials)
  - `docs/troubleshooting.md` (~555 lines)
- **File refactors**: streaming, query_expansion, circuit_breaker, connection_pool — all sub-files ≤ 554 lines

### Codebase Statistics (v0.5.0)
- **Source Files**: ~109 Rust files (+1: `src/bin/oxirag-server.rs`, +1 test: `tests/trait_bounds_native.rs`)
- **Total Lines**: ~73,000+ (Rust code)
- **Tests**: 1,658 + new trait-bound compile tests
- **Clippy Warnings**: 0
- **Rustdoc Warnings**: 0
- **New Features (v0.5.0)**: `python` (PyO3 bindings), `wasm-indexeddb` (IndexedDB VectorStore), `wasm-prefix-indexeddb` (IndexedDB PrefixCache)
- **New Files**: `src/bin/oxirag-server.rs`, `Dockerfile`, `.dockerignore`, `docker-compose.yml`, `docs/docker.md`, `tests/trait_bounds_native.rs`
- **Trait Refactor**: `VectorStore`, `Echo`, `EmbeddingProvider`, `MultiModalEmbeddingProvider`, `PrefixCacheStore`, `PrefixCacheExt` now use cfg-gated `async_trait(?Send)` on wasm32

### Codebase Statistics (v0.4.0)
- **Source Files**: 102 Rust files (+3: `observability/otel.rs`, `layer1_echo/embedding/clip.rs`, `rest_server.rs`)
- **Total Lines**: ~73,000 (Rust code)
- **Tests**: 1,658 (+89 from v0.3.0: 5 observer, 7 OTel, 25+ CLIP/multimodal, 19 REST, 12 integration E2E, 21 proptest fuzz)
- **Clippy Warnings**: 0
- **Rustdoc Warnings**: 0
- **New Features (v0.4.0)**: `otel` (OpenTelemetry OTLP+stdout), `multimodal` (CLIP text+image), `rest-server` (axum HTTP API), SpanObserver/MemoryObserver, cross-layer integration test suite, proptest fuzz harness
- **New Features (v0.3.0)**: RedbGraphStore (feature `graphrag-redb`), HiddenStatePooling + apply\_hidden\_state\_pooling, extract\_sentence\_embedding, PipelineSpanContext / LayerSpan / SpanReport observability module
- **New Features (v0.2.0)**: RedbPrefixCache (feature `prefix-cache-redb`), RedbVectorStore (feature `echo-redb`), CandleHiddenStateProvider (features `hidden-states+speculator`), BGE/MPNet embedding presets
- **Refactored (v0.2.0)**: `distillation/progressive.rs` (1741 lines) → `distillation/progressive/` (4 files, all under 700 lines)
- **Performance**: 5.6x-9.0x faster similarity computations with SIMD (unchanged)

### Codebase Statistics (v0.3.0)
- **Source Files**: 99 Rust files (+2 from v0.2.0: `layer4_graph/redb_store.rs`, `observability.rs`)
- **Total Lines**: ~68,000 (Rust code)
- **Tests**: 1,569 (+65 from v0.2.0: 12 RedbGraphStore, 12 observability, 8 prefix_cache proptest, 8 echo-redb proptest, 7 embedding preset, 9 hidden-state pooling, 9 more candle_provider)
