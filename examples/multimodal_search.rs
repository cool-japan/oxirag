//! Multi-modal (text + image) semantic search example.
//!
//! Demonstrates building a multi-modal Echo layer using a `MockMultiModalProvider`
//! that generates deterministic embeddings without downloading any models.
//!
//! # Running
//!
//! ```bash
//! cargo run --example multimodal_search --features multimodal,native
//! ```
//!
//! # Architecture note
//! In production you would replace `MockMultiModalProvider` with
//! `CandleClipProvider` (requires `--features multimodal` and a network download):
//!
//! ```rust,ignore
//! use oxirag::layer1_echo::{CandleClipProvider, ClipPreset};
//! let provider = CandleClipProvider::new(ClipPreset::VitBase32)?;
//! ```

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use async_trait::async_trait;
use oxirag::layer1_echo::{
    Echo, EchoLayer, EmbeddingInput, InMemoryVectorStore, MultiModalEmbeddingProvider, VectorStore,
};
use oxirag::types::Document;

// ─────────────────────────────────────────────────────────────────────────────
// Mock multi-modal provider (no model download required)
// ─────────────────────────────────────────────────────────────────────────────

/// A deterministic mock embedding provider that handles text, images, and
/// joint text+image inputs using stable hash-derived unit vectors.
///
/// This is intentionally simple — it exists so the example compiles and runs
/// in CI without downloading CLIP weights.  Embeddings are semantically
/// meaningless but structurally correct (unit vectors of the declared dimension).
struct MockMultiModalProvider {
    dim: usize,
}

impl MockMultiModalProvider {
    fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Generate a deterministic unit vector from an arbitrary byte slice.
    fn bytes_to_unit_vec(&self, seed: &[u8]) -> Vec<f32> {
        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        let mut state = hasher.finish();

        let mut vec: Vec<f32> = (0..self.dim)
            .map(|_| {
                #[allow(clippy::cast_precision_loss)]
                let v = ((state % 10_000) as f32 / 5_000.0) - 1.0;
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                v
            })
            .collect();

        // L2 normalise
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-9 {
            for v in &mut vec {
                *v /= norm;
            }
        }
        vec
    }

    /// Average two unit vectors and re-normalise.
    fn average_unit_vecs(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut avg: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| (x + y) * 0.5).collect();
        let norm: f32 = avg.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-9 {
            for v in &mut avg {
                *v /= norm;
            }
        }
        avg
    }
}

#[async_trait]
impl MultiModalEmbeddingProvider for MockMultiModalProvider {
    async fn embed_multi(
        &self,
        input: EmbeddingInput<'_>,
    ) -> Result<Vec<f32>, oxirag::error::EmbeddingError> {
        match input {
            EmbeddingInput::Text(text) => Ok(self.bytes_to_unit_vec(text.as_bytes())),
            EmbeddingInput::Image(bytes) => Ok(self.bytes_to_unit_vec(bytes)),
            EmbeddingInput::TextAndImage { text, image } => {
                let text_vec = self.bytes_to_unit_vec(text.as_bytes());
                let img_vec = self.bytes_to_unit_vec(image);
                Ok(self.average_unit_vecs(&text_vec, &img_vec))
            }
        }
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn model_id(&self) -> &str {
        "mock-multimodal-provider"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "native")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OxiRAG Multi-Modal Search Example ===\n");

    const DIM: usize = 512;

    // Build the Echo layer with a mock multi-modal provider.
    // In production, swap `MockMultiModalProvider` for `CandleClipProvider`.
    let provider = MockMultiModalProvider::new(DIM);
    let store = InMemoryVectorStore::new(DIM);
    let mut echo = EchoLayer::new(provider, store);

    // ── Index five fake "images" described by their filenames ─────────────
    let image_docs = [
        ("photo_cat.jpg", "A domestic cat sitting on a sofa"),
        ("photo_dog.jpg", "A golden retriever playing in the park"),
        ("photo_car.jpg", "A red sports car on a race track"),
        ("photo_tree.jpg", "A tall oak tree in autumn"),
        ("photo_sunset.jpg", "A vibrant sunset over the ocean"),
    ];

    println!("Indexing documents...");
    for (filename, description) in &image_docs {
        // We simulate an "image document" by storing the filename as content
        // and the human-readable description as metadata.
        let doc = Document::new(*filename).with_metadata("description", *description);
        let id = echo.index(doc).await?;
        println!("  Indexed: {filename} (id: {id})");
    }

    println!("\nStore contains {} document(s).\n", echo.count().await);

    // ── Perform a text-query search ───────────────────────────────────────
    let query = "animal companion";
    println!("Text query: \"{query}\"");
    let results = echo.search(query, 3, None).await?;
    println!("Top {} results:", results.len());
    for (rank, result) in results.iter().enumerate() {
        println!(
            "  #{}: {} (score: {:.4})",
            rank + 1,
            result.document.content,
            result.score
        );
        if let Some(desc) = result.document.metadata.get("description") {
            println!("      description: {desc}");
        }
    }

    // ── Demonstrate joint text+image embedding ────────────────────────────
    println!("\nDemonstrating joint text+image embedding...");
    let fake_image_bytes: Vec<u8> = (0_u8..=255).cycle().take(1024).collect();
    let joint_query_text = "outdoor nature scene";

    // The `EmbeddingProvider` blanket impl routes Text variants through embed()
    // We access the multi-modal path directly via the provider API here.
    let joint_embedding = {
        // Access the embedding provider through a temporary mock for demonstration
        let demo_provider = MockMultiModalProvider::new(DIM);
        demo_provider
            .embed_multi(EmbeddingInput::TextAndImage {
                text: joint_query_text,
                image: &fake_image_bytes,
            })
            .await?
    };

    // Verify the joint embedding is a unit vector
    let norm: f32 = joint_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!(
        "Joint embedding: dim={}, L2-norm={:.6} (expected ~1.0)",
        joint_embedding.len(),
        norm
    );
    assert!(
        (norm - 1.0).abs() < 1e-4,
        "Joint embedding is not unit-normalised: norm={norm}"
    );

    // ── Search using the joint embedding directly ─────────────────────────
    println!("\nSearching with joint text+image embedding for \"{joint_query_text}\"...");
    let store_ref = echo.vector_store();
    let joint_results = store_ref.search(&joint_embedding, 3, None).await?;
    println!("Top {} results:", joint_results.len());
    for (rank, result) in joint_results.iter().enumerate() {
        println!(
            "  #{}: {} (score: {:.4})",
            rank + 1,
            result.document.content,
            result.score
        );
    }

    println!("\nExample completed successfully.");
    Ok(())
}

#[cfg(not(feature = "native"))]
fn main() {
    eprintln!(
        "This example requires the `native` feature. \
        Run: cargo run --example multimodal_search --features multimodal,native"
    );
}
