//! Shared test-only fixtures: document builders and deterministic synthetic
//! text generators used across the `corpus_curation` test suite.
//!
//! Every vocabulary below is disjoint from the others so that different
//! tests (classifier, contamination, near-duplicate) never accidentally
//! share content and contaminate each other's measurements.

use crate::corpus_curation::CurationRng;
use crate::types::Document;

/// Build a `Document` with an explicit id.
pub(super) fn doc(id: &str, content: impl Into<String>) -> Document {
    Document::new(content).with_id(id)
}

/// Distinctive, non-stop-word vocabulary for "good" documents (classifier
/// positive class; ablation on-topic prose). Includes "quantum", used by the
/// ablation test's query.
pub(super) const GOOD_VOCAB: &[&str] = &[
    "quantum",
    "photon",
    "lattice",
    "entropy",
    "gradient",
    "manifold",
    "electron",
    "orbital",
    "resonance",
    "spectrum",
    "isotope",
    "catalyst",
    "polymer",
    "membrane",
    "enzyme",
    "neuron",
    "synapse",
    "cortex",
    "algorithm",
    "theorem",
    "integral",
    "vector",
    "tensor",
    "circuit",
    "voltage",
    "current",
    "magnetic",
    "thermal",
    "kinetic",
    "velocity",
];

/// Low-information, spam-like vocabulary for "bad" documents (classifier
/// negative class), visibly disjoint from [`GOOD_VOCAB`].
pub(super) const BAD_VOCAB: &[&str] = &[
    "buy",
    "cheap",
    "free",
    "click",
    "now",
    "win",
    "prize",
    "offer",
    "deal",
    "sale",
    "cash",
    "bonus",
    "urgent",
    "limited",
    "act",
    "today",
    "guaranteed",
    "amazing",
    "shocking",
    "viral",
];

/// Vocabulary for held-out benchmark items and their planted copies
/// (contamination tests).
pub(super) const BENCH_VOCAB: &[&str] = &[
    "glacier",
    "obsidian",
    "cathedral",
    "tapestry",
    "meridian",
    "harbor",
    "lantern",
    "orchard",
    "quarry",
    "citadel",
    "reservoir",
    "aqueduct",
    "embassy",
    "pavilion",
    "causeway",
    "rampart",
    "granary",
    "observatory",
    "monastery",
    "vineyard",
    "foundry",
    "archive",
    "conservatory",
    "plateau",
];

/// Vocabulary for "clean" corpus filler text that shares no words with
/// [`BENCH_VOCAB`] (contamination tests) or [`GOOD_VOCAB`]/[`BAD_VOCAB`].
pub(super) const CLEAN_VOCAB: &[&str] = &[
    "bicycle",
    "umbrella",
    "kitchen",
    "calendar",
    "notebook",
    "blanket",
    "sandwich",
    "balloon",
    "sidewalk",
    "hallway",
    "mirror",
    "cabinet",
    "curtain",
    "doorbell",
    "pillow",
    "faucet",
    "staircase",
    "fireplace",
    "mailbox",
    "driveway",
];

/// Vocabulary for near-duplicate-clustering fixtures.
pub(super) const NEARDUP_VOCAB: &[&str] = &[
    "aurora", "basalt", "cascade", "delta", "ember", "fjord", "glimmer", "horizon", "islet",
    "jungle", "kelp", "lagoon", "mesa", "nebula", "oasis", "prairie", "quartz", "ridge", "summit",
    "tundra",
];

/// Common English connective words, deliberately overlapping
/// [`crate::corpus_curation::heuristics`]'s stop-word list, shared by every
/// vocabulary above so that separating classes requires weighing multiple
/// hashed features rather than detecting the presence of any single word.
pub(super) const CONNECTIVES: &[&str] = &[
    "the", "a", "of", "in", "and", "is", "to", "with", "for", "on", "this", "that", "as", "was",
];

/// Draw a `length`-word pseudo-sentence: roughly one word in three is a
/// [`CONNECTIVES`] filler word, the rest are drawn uniformly from `vocab`.
/// Always returns exactly `length` words.
pub(super) fn draw_text(rng: &mut CurationRng, vocab: &[&str], length: usize) -> String {
    let mut words: Vec<&str> = Vec::with_capacity(length);
    for _ in 0..length {
        if rng.next_bool(0.35)
            && let Some(w) = rng.choose(CONNECTIVES)
        {
            words.push(w);
            continue;
        }
        if let Some(w) = rng.choose(vocab) {
            words.push(w);
        }
    }
    words.join(" ")
}
