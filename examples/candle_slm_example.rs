//! Example demonstrating CandleSLM usage with Phi-2 model.
//!
//! This example shows how to use the real Candle-based SLM implementation
//! for text generation and verification.
//!
//! # Running the example
//!
//! ```bash
//! cargo run --example candle_slm_example --features speculator,native
//! ```
//!
//! # Note
//!
//! - First run will download Phi-2 model (~2.7GB) from HuggingFace
//! - Model is cached in ~/.cache/huggingface/hub/
//! - This example is CPU-only by default
//! - For GPU support, add --features cuda or --features metal

use oxirag::layer2_speculator::{
    CandleSLM, CandleSlmConfig, CandleSlmDevice, FinishReason, SlmConfig, SmallLanguageModel,
};

#[cfg(feature = "native")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OxiRAG Candle SLM Example ===\n");

    // Configure the SLM with Phi-2 model
    println!("Initializing Candle SLM with Phi-2 model...");
    println!("Note: First run will download the model (~2.7GB)\n");

    let candle_config = CandleSlmConfig {
        model_id: "microsoft/phi-2".to_string(),
        revision: "main".to_string(),
        device: CandleSlmDevice::Cpu,
        speculator_config: Default::default(),
    };

    // Create the SLM instance
    let slm = match CandleSLM::new(candle_config) {
        Ok(slm) => {
            println!("Model loaded successfully!");
            println!("Device: {:?}", slm.device());
            println!("Model info: {:?}\n", slm.model_info());
            slm
        }
        Err(e) => {
            eprintln!("Failed to load model: {}", e);
            eprintln!("\nTroubleshooting:");
            eprintln!("1. Check internet connection (model download required)");
            eprintln!("2. Ensure sufficient disk space (~3GB)");
            eprintln!("3. Check HuggingFace Hub access");
            return Err(e.into());
        }
    };

    // Example 1: Basic text generation
    println!("--- Example 1: Basic Text Generation ---");
    let prompt = "What is the capital of France?";
    println!("Prompt: {}", prompt);

    let gen_config = SlmConfig::new("microsoft/phi-2")
        .with_max_tokens(128)
        .with_temperature(0.3)
        .with_top_p(0.9);

    let output = slm.generate(prompt, &gen_config).await?;
    println!("Generated text: {}", output.text);
    println!("Token count: {}", output.tokens.len());
    println!(
        "Finish reason: {}",
        match output.finish_reason {
            FinishReason::Stop => "EOS token",
            FinishReason::MaxTokens => "Max tokens reached",
            FinishReason::Error(ref e) => e,
        }
    );
    if let Some(ref logprobs) = output.logprobs {
        println!(
            "Average log probability: {:.3}",
            logprobs.iter().sum::<f32>() / logprobs.len() as f32
        );
    }
    println!();

    // Example 2: Verification
    println!("--- Example 2: Text Verification ---");
    let context = "Paris is the capital and most populous city of France. It has been one of Europe's major centers of finance, diplomacy, commerce, fashion, gastronomy, science, and arts.";
    let draft = "Paris is the capital of France.";

    println!("Context: {}", context);
    println!("Draft: {}", draft);

    let confidence = slm.verify_text(draft, context).await?;
    println!("Verification confidence: {:.2}%", confidence * 100.0);
    println!();

    // Example 3: Log probabilities
    println!("--- Example 3: Computing Log Probabilities ---");
    let text = "Hello, world!";
    println!("Text: {}", text);

    let logprobs = slm.get_logprobs(text).await?;
    println!("Log probabilities:");
    for (i, logprob) in logprobs.iter().enumerate() {
        println!("  Token {}: {:.4}", i, logprob);
    }
    println!();

    // Example 4: Different prompts
    println!("--- Example 4: Multiple Prompts ---");
    let prompts = vec![
        "Explain quantum computing in one sentence:",
        "What is the largest planet in our solar system?",
        "Define machine learning:",
    ];

    for (i, prompt) in prompts.iter().enumerate() {
        println!("Prompt {}: {}", i + 1, prompt);

        let config = SlmConfig::new("microsoft/phi-2")
            .with_max_tokens(64)
            .with_temperature(0.5);

        match slm.generate(prompt, &config).await {
            Ok(output) => {
                println!("Response: {}", output.text.trim());
            }
            Err(e) => {
                println!("Error: {}", e);
            }
        }
        println!();
    }

    println!("=== Example completed successfully! ===");
    Ok(())
}

#[cfg(not(feature = "native"))]
fn main() {
    eprintln!("This example requires the 'native' feature to be enabled.");
    eprintln!("Run with: cargo run --example candle_slm_example --features speculator,native");
    std::process::exit(1);
}
