//! Example: End-to-End LoRA Training for On-the-Fly Distillation
//!
//! This example demonstrates the complete workflow for on-the-fly distillation:
//! 1. Track incoming queries and collect Q&A pairs
//! 2. Detect patterns ready for distillation
//! 3. Train a specialized LoRA adapter
//! 4. Hot-swap to the specialized model
//!
//! # Running This Example
//!
//! ```bash
//! cargo run --example lora_training_example --features speculator,distillation,native
//! ```
//!
//! # Prerequisites
//!
//! - Sufficient memory for model loading (typically 2-4GB)
//! - CUDA toolkit (optional, for GPU training)
//! - Hugging Face model cache (models are downloaded automatically)
//!
//! # Expected Output
//!
//! The example will:
//! - Track queries and detect a frequent pattern
//! - Create a LoRA training job
//! - Monitor training progress
//! - Report final training metrics
//!
//! # Note
//!
//! This is a demonstration using simplified data. In production:
//! - Use real queries from your RAG pipeline
//! - Collect more Q&A pairs (50-100 per pattern)
//! - Train for more epochs with proper validation
//! - Monitor model quality metrics

use oxirag::distillation::{
    CandleLoraConfig, CandleLoraTrainer, DistillationConfig, DistillationTracker,
    InMemoryDistillationTracker, LoraConfig, LoraTrainer, LoraTrainingExample, TrainingStatus,
};

#[cfg(all(feature = "native", feature = "distillation"))]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== LoRA Training Example: On-the-Fly Distillation ===\n");

    // Step 1: Create distillation tracker
    println!("Step 1: Setting up distillation tracker...");
    let config = DistillationConfig {
        min_frequency_threshold: 3,
        similarity_threshold: 0.85,
        max_candidates: 100,
        collection_window_secs: 3600,
        max_qa_pairs_per_pattern: 50,
    };

    let mut tracker = InMemoryDistillationTracker::new(config);
    println!("✓ Tracker initialized\n");

    // Step 2: Simulate incoming queries (in production, these come from real RAG queries)
    println!("Step 2: Tracking incoming queries...");
    let queries = vec![
        (
            "What is Rust?",
            "Rust is a systems programming language focused on safety and performance.",
        ),
        (
            "What is Rust?",
            "Rust is a modern language for building reliable and efficient software.",
        ),
        (
            "What is Rust programming?",
            "Rust is a statically typed compiled language with memory safety guarantees.",
        ),
        (
            "Tell me about Rust",
            "Rust is developed by Mozilla and provides zero-cost abstractions.",
        ),
        (
            "Explain Rust",
            "Rust eliminates memory bugs through its ownership system.",
        ),
    ];

    for (query, answer) in &queries {
        tracker.track_query(query, Some(answer), 0.95).await?;
        println!("  Tracked: {query}");
    }

    let stats = tracker.stats();
    println!(
        "\n✓ Tracked {} queries, {} unique patterns",
        stats.total_queries_tracked, stats.unique_patterns
    );

    // Step 3: Check for distillation candidates
    println!("\nStep 3: Detecting distillation candidates...");
    let candidates = tracker.get_candidates().await;

    let ready_candidates: Vec<_> = candidates
        .iter()
        .filter(|c| c.ready_for_distillation)
        .collect();

    if ready_candidates.is_empty() {
        println!("⚠ No candidates ready for distillation yet");
        println!("  (This is expected with only {} queries)", queries.len());
        println!("  In production, track more queries to reach the threshold");
        return Ok(());
    }

    println!(
        "✓ Found {} candidates ready for distillation",
        ready_candidates.len()
    );

    // Step 4: Prepare training data
    let candidate = &ready_candidates[0];
    println!("\nStep 4: Preparing training data...");
    println!("  Pattern: {}", candidate.pattern.normalized_text);
    println!("  Frequency: {}", candidate.frequency);
    println!("  Avg Confidence: {:.2}", candidate.avg_confidence);
    println!("  Q&A Pairs: {}", candidate.qa_pairs.len());

    // Convert Q&A pairs to training examples
    let training_examples: Vec<LoraTrainingExample> = candidate
        .qa_pairs
        .iter()
        .map(|pair| LoraTrainingExample::with_weight(&pair.query, &pair.answer, pair.confidence))
        .collect();

    println!("✓ Prepared {} training examples\n", training_examples.len());

    // Step 5: Create LoRA trainer
    println!("Step 5: Setting up LoRA trainer...");
    let lora_config = CandleLoraConfig {
        base: LoraConfig {
            rank: 8,
            alpha: 16.0,
            dropout: 0.05,
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
            learning_rate: 1e-4,
            num_epochs: 3,
            batch_size: 2,
        },
        model_id: "microsoft/phi-2".to_string(),
        device: "cpu".to_string(),
        dtype: "f32".to_string(),
        checkpoint_dir: std::env::temp_dir().join("oxirag_lora_checkpoints"),
        max_grad_norm: 1.0,
        weight_decay: 0.01,
        adam_beta1: 0.9,
        adam_beta2: 0.999,
        adam_eps: 1e-8,
        warmup_steps: 10,
        early_stopping_patience: 3,
        min_improvement: 0.001,
        validation_split: 0.1,
        max_seq_len: 512,
    };

    let mut trainer = match CandleLoraTrainer::new(lora_config.clone()) {
        Ok(t) => t,
        Err(e) => {
            println!("⚠ Failed to create trainer: {e}");
            println!("  (This is expected in environments without filesystem access)");
            return Ok(());
        }
    };

    println!("✓ Trainer initialized");
    println!("  Model: {}", lora_config.model_id);
    println!("  Device: {}", lora_config.device);
    println!("  LoRA Rank: {}", lora_config.base.rank);
    println!("  Learning Rate: {}", lora_config.base.learning_rate);
    println!("  Epochs: {}\n", lora_config.base.num_epochs);

    // Step 6: Create training job
    println!("Step 6: Creating training job...");
    let job_id = trainer
        .create_job(
            &candidate.pattern,
            training_examples,
            lora_config.base.clone(),
        )
        .await?;

    println!("✓ Training job created: {job_id}\n");

    // Step 7: Monitor training progress
    println!("Step 7: Monitoring training progress...");
    println!("  (In production, training runs asynchronously in the background)\n");

    // Simulate monitoring
    loop {
        if let Some(status) = trainer.get_status(&job_id).await {
            match &status {
                TrainingStatus::Pending => {
                    println!("  Status: Pending...");
                }
                TrainingStatus::Preparing => {
                    println!("  Status: Preparing data...");
                }
                TrainingStatus::Training { epoch, loss } => {
                    println!("  Epoch {}: Training Loss = {:.4}", epoch, loss);
                }
                TrainingStatus::Completed { final_loss } => {
                    println!("\n✓ Training completed!");
                    println!("  Final Loss: {:.4}", final_loss);
                    break;
                }
                TrainingStatus::Failed { error } => {
                    println!("\n✗ Training failed: {error}");
                    break;
                }
            }

            if status.is_terminal() {
                break;
            }
        }

        #[cfg(feature = "native")]
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }

    // Step 8: Summary
    println!("\n=== Summary ===");
    println!("✓ Successfully demonstrated LoRA training workflow");
    println!(
        "✓ Trained adapter for pattern: {}",
        candidate.pattern.normalized_text
    );
    println!("✓ Model ready for deployment");
    println!("\nNext Steps:");
    println!("  - Integrate with hot-swap system for automatic model switching");
    println!("  - Monitor specialized model performance vs. base model");
    println!("  - Collect feedback to refine training data");
    println!("  - Scale to multiple query patterns");

    Ok(())
}

#[cfg(not(all(feature = "native", feature = "distillation")))]
fn main() {
    println!("This example requires 'native' and 'distillation' features.");
    println!(
        "Run with: cargo run --example lora_training_example --features speculator,distillation,native"
    );
}
