//! Tests for the `conversation` module.
//!
//! All async tests use `#[tokio::test]`. No file I/O is performed.

use std::sync::Arc;

use super::buffer::{
    FullHistoryBuffer, HistoryBuffer, HybridBuffer, SlidingWindowBuffer, SummaryBuffer,
};
use super::reformulator::{
    ConversationAwareQuery, FollowUpDetector, QueryReformulator, ReformulationStrategy,
};
use super::session::{ConversationalPipeline, InMemorySessionManager, SessionManager};
use super::types::{
    ConversationError, ConversationHistory, ConversationId, Session, SessionConfig, Turn, TurnRole,
};

// ── helpers ───────────────────────────────────────────────────────────────────

fn make_history(pairs: &[(&str, &str)]) -> ConversationHistory {
    let mut h = ConversationHistory::new();
    for (user, asst) in pairs {
        h.add_turn(Turn::new(TurnRole::User, *user));
        h.add_turn(Turn::new(TurnRole::Assistant, *asst));
    }
    h
}

// ── Turn ──────────────────────────────────────────────────────────────────────

#[test]
fn test_turn_new_user() {
    let t = Turn::new(TurnRole::User, "Hello");
    assert!(t.is_user());
    assert!(!t.is_assistant());
    assert_eq!(t.text, "Hello");
    assert!(t.metadata.is_empty());
}

#[test]
fn test_turn_new_assistant() {
    let t = Turn::new(TurnRole::Assistant, "Hi!");
    assert!(t.is_assistant());
    assert!(!t.is_user());
}

#[test]
fn test_turn_with_metadata() {
    let t = Turn::new(TurnRole::User, "q")
        .with_metadata("latency_ms", "42")
        .with_metadata("model", "test");
    assert_eq!(t.metadata.get("latency_ms").map(String::as_str), Some("42"));
    assert_eq!(t.metadata.get("model").map(String::as_str), Some("test"));
}

#[test]
fn test_turn_role_display() {
    assert_eq!(TurnRole::User.to_string(), "User");
    assert_eq!(TurnRole::Assistant.to_string(), "Assistant");
}

// ── ConversationHistory ───────────────────────────────────────────────────────

#[test]
fn test_history_starts_empty() {
    let h = ConversationHistory::new();
    assert!(h.is_empty());
    assert_eq!(h.len(), 0);
}

#[test]
fn test_history_add_and_len() {
    let mut h = ConversationHistory::new();
    h.add_turn(Turn::new(TurnRole::User, "a"));
    h.add_turn(Turn::new(TurnRole::Assistant, "b"));
    assert_eq!(h.len(), 2);
    assert!(!h.is_empty());
}

#[test]
fn test_history_last_user_turn() {
    let mut h = ConversationHistory::new();
    h.add_turn(Turn::new(TurnRole::User, "first"));
    h.add_turn(Turn::new(TurnRole::Assistant, "reply"));
    h.add_turn(Turn::new(TurnRole::User, "second"));
    assert_eq!(h.last_user_turn().map(|t| t.text.as_str()), Some("second"));
}

#[test]
fn test_history_last_assistant_turn() {
    let mut h = ConversationHistory::new();
    h.add_turn(Turn::new(TurnRole::User, "q"));
    h.add_turn(Turn::new(TurnRole::Assistant, "a1"));
    h.add_turn(Turn::new(TurnRole::Assistant, "a2"));
    assert_eq!(h.last_assistant_turn().map(|t| t.text.as_str()), Some("a2"));
}

#[test]
fn test_history_last_turns_when_empty() {
    let h = ConversationHistory::new();
    assert!(h.last_user_turn().is_none());
    assert!(h.last_assistant_turn().is_none());
}

#[test]
fn test_history_recent_turns_fewer_than_n() {
    let mut h = ConversationHistory::new();
    h.add_turn(Turn::new(TurnRole::User, "x"));
    let recent = h.recent_turns(10);
    assert_eq!(recent.len(), 1);
}

#[test]
fn test_history_recent_turns_exact_n() {
    let h = make_history(&[("q1", "a1"), ("q2", "a2")]);
    let recent = h.recent_turns(4);
    assert_eq!(recent.len(), 4);
}

#[test]
fn test_history_recent_turns_more_than_n() {
    let h = make_history(&[("q1", "a1"), ("q2", "a2"), ("q3", "a3")]);
    let recent = h.recent_turns(2);
    assert_eq!(recent.len(), 2);
    assert_eq!(recent[0].text, "q3");
    assert_eq!(recent[1].text, "a3");
}

#[test]
fn test_history_iter_order() {
    let h = make_history(&[("u1", "a1"), ("u2", "a2")]);
    let texts: Vec<&str> = h.iter().map(|t| t.text.as_str()).collect();
    assert_eq!(texts, ["u1", "a1", "u2", "a2"]);
}

#[test]
fn test_history_truncate_to() {
    let mut h = make_history(&[("u1", "a1"), ("u2", "a2"), ("u3", "a3")]);
    // 6 turns initially
    h.truncate_to(3);
    assert_eq!(h.len(), 3);
    // oldest 3 removed — remaining should be u2, a2, u3, a3... wait, that's 4
    // actually oldest 3 = u1, a1, u2 → remaining = a2, u3, a3
    assert_eq!(h.as_slice()[0].text, "a2");
}

// ── ConversationId ────────────────────────────────────────────────────────────

#[test]
fn test_conversation_id_new_is_unique() {
    let id1 = ConversationId::new();
    let id2 = ConversationId::new();
    assert_ne!(id1, id2);
}

#[test]
fn test_conversation_id_as_str_is_uuid_format() {
    let id = ConversationId::new();
    let s = id.as_str();
    // UUID hyphenated format: 8-4-4-4-12 chars = 36 total
    assert_eq!(s.len(), 36);
    assert_eq!(s.chars().filter(|&c| c == '-').count(), 4);
}

#[test]
fn test_conversation_id_display() {
    let id = ConversationId::new();
    assert_eq!(id.to_string(), id.as_str());
}

// ── SessionConfig ─────────────────────────────────────────────────────────────

#[test]
fn test_session_config_defaults() {
    let cfg = SessionConfig::default();
    assert_eq!(cfg.max_turns, 100);
    assert_eq!(cfg.summary_threshold, 10);
}

#[test]
fn test_session_config_builder() {
    let cfg = SessionConfig::default()
        .with_max_turns(50)
        .with_summary_threshold(5);
    assert_eq!(cfg.max_turns, 50);
    assert_eq!(cfg.summary_threshold, 5);
}

// ── Session ───────────────────────────────────────────────────────────────────

#[test]
fn test_session_new() {
    let s = Session::new(SessionConfig::default());
    assert!(s.history.is_empty());
    assert_eq!(s.turn_count(), 0);
}

#[test]
fn test_session_add_turn_enforces_max() {
    let cfg = SessionConfig::default().with_max_turns(3);
    let mut s = Session::new(cfg);
    for i in 0..5 {
        s.add_turn(Turn::new(TurnRole::User, format!("q{i}")));
    }
    assert_eq!(s.turn_count(), 3);
}

#[test]
fn test_session_with_id() {
    let id = ConversationId::new();
    let s = Session::with_id(id.clone(), SessionConfig::default());
    assert_eq!(s.id, id);
}

// ── FullHistoryBuffer ─────────────────────────────────────────────────────────

#[tokio::test]
async fn test_full_history_buffer_empty() {
    let h = ConversationHistory::new();
    let buf = FullHistoryBuffer;
    let ctx = buf.get_context(&h).await;
    assert!(ctx.is_empty());
}

#[tokio::test]
async fn test_full_history_buffer_all_turns() {
    let h = make_history(&[("Hi", "Hello!"), ("What is Rust?", "A systems language.")]);
    let buf = FullHistoryBuffer;
    let ctx = buf.get_context(&h).await;
    assert!(ctx.contains("User: Hi"));
    assert!(ctx.contains("Assistant: Hello!"));
    assert!(ctx.contains("User: What is Rust?"));
    assert!(ctx.contains("Assistant: A systems language."));
}

// ── SlidingWindowBuffer ───────────────────────────────────────────────────────

#[tokio::test]
async fn test_sliding_window_buffer_smaller_than_window() {
    let h = make_history(&[("q", "a")]);
    let buf = SlidingWindowBuffer::new(10);
    let ctx = buf.get_context(&h).await;
    assert!(ctx.contains("User: q"));
    assert!(ctx.contains("Assistant: a"));
}

#[tokio::test]
async fn test_sliding_window_buffer_larger_than_window() {
    let h = make_history(&[("q1", "a1"), ("q2", "a2"), ("q3", "a3"), ("q4", "a4")]);
    let buf = SlidingWindowBuffer::new(2); // Only last 2 turns
    let ctx = buf.get_context(&h).await;
    assert!(!ctx.contains("q1"));
    assert!(!ctx.contains("a1"));
    assert!(ctx.contains("q4"));
    assert!(ctx.contains("a4"));
}

#[tokio::test]
async fn test_sliding_window_default() {
    let buf = SlidingWindowBuffer::default();
    assert_eq!(buf.window_size, 6);
}

// ── SummaryBuffer ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_summary_buffer_small_history() {
    let h = make_history(&[("q1", "a1")]);
    let buf = SummaryBuffer::new(4, 500);
    let ctx = buf.get_context(&h).await;
    assert!(ctx.contains("User: q1"));
    assert!(ctx.contains("Assistant: a1"));
}

#[tokio::test]
async fn test_summary_buffer_large_history() {
    let mut h = ConversationHistory::new();
    for i in 0..20 {
        h.add_turn(Turn::new(TurnRole::User, format!("question {i}")));
        h.add_turn(Turn::new(TurnRole::Assistant, format!("answer {i}")));
    }
    let buf = SummaryBuffer::new(4, 500);
    let ctx = buf.get_context(&h).await;
    // Should contain recent turns.
    assert!(ctx.contains("question 19"));
    // Should contain summary header for older turns.
    assert!(ctx.contains("Summary of earlier"));
}

#[tokio::test]
async fn test_summary_buffer_truncates_summary() {
    let mut h = ConversationHistory::new();
    for i in 0..50 {
        h.add_turn(Turn::new(
            TurnRole::User,
            format!("longerquestion{i}_aaaa_bbbb_cccc_dddd_eeee"),
        ));
        h.add_turn(Turn::new(
            TurnRole::Assistant,
            format!("longeranswer{i}_xxxx_yyyy_zzzz_0000_1111"),
        ));
    }
    let buf = SummaryBuffer::new(2, 100);
    let ctx = buf.get_context(&h).await;
    // The summary section must not exceed max_summary_chars by a large margin.
    let summary_section = ctx.lines().next().unwrap_or("");
    assert!(
        summary_section.len() < 200,
        "Summary section too long: {}",
        summary_section.len()
    );
}

// ── HybridBuffer ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_hybrid_buffer_empty() {
    let h = ConversationHistory::new();
    let buf = HybridBuffer::new(4, 300);
    let ctx = buf.get_context(&h).await;
    assert!(ctx.is_empty());
}

#[tokio::test]
async fn test_hybrid_buffer_within_recent() {
    let h = make_history(&[("q", "a")]);
    let buf = HybridBuffer::new(6, 300);
    let ctx = buf.get_context(&h).await;
    assert!(ctx.contains("User: q"));
}

#[tokio::test]
async fn test_hybrid_buffer_beyond_recent() {
    let h = make_history(&[
        ("q1", "a1"),
        ("q2", "a2"),
        ("q3", "a3"),
        ("q4", "a4"),
        ("q5", "a5"),
    ]);
    let buf = HybridBuffer::new(2, 300);
    let ctx = buf.get_context(&h).await;
    assert!(ctx.contains("q5"));
    assert!(ctx.contains("Earlier context"));
    assert!(!ctx.contains("User: q1")); // q1 is in the summary, not verbatim
}

// ── FollowUpDetector::is_follow_up ────────────────────────────────────────────

#[test]
fn test_follow_up_short_query() {
    assert!(FollowUpDetector::is_follow_up("Why?"));
    assert!(FollowUpDetector::is_follow_up("Tell me more."));
}

#[test]
fn test_follow_up_with_pronoun_it() {
    assert!(FollowUpDetector::is_follow_up(
        "Can you explain how it is designed to handle this issue?"
    ));
}

#[test]
fn test_follow_up_with_pronoun_they() {
    assert!(FollowUpDetector::is_follow_up(
        "Where did they come from originally in the discussion?"
    ));
}

#[test]
fn test_follow_up_with_pronoun_this() {
    assert!(FollowUpDetector::is_follow_up(
        "Can you clarify this particular aspect of the design?"
    ));
}

#[test]
fn test_follow_up_with_qualifier_also() {
    assert!(FollowUpDetector::is_follow_up(
        "What are the performance implications also?"
    ));
}

#[test]
fn test_follow_up_with_qualifier_too() {
    assert!(FollowUpDetector::is_follow_up(
        "Does it apply to async code too?"
    ));
}

#[test]
fn test_not_follow_up_standalone() {
    // A long standalone query with no pronouns or qualifiers.
    let standalone =
        "What are the performance characteristics of B-tree indexes in PostgreSQL databases?";
    assert!(!FollowUpDetector::is_follow_up(standalone));
}

#[test]
fn test_follow_up_pronoun_word_boundary() {
    // "iteration" contains "it" but not at a word boundary after "it".
    // The word boundary check should prevent "iteration" from matching.
    let query = "What are the iteration performance characteristics in this particular codebase?";
    // "this" IS a pronoun at word boundary, so it IS a follow-up.
    assert!(FollowUpDetector::is_follow_up(query));

    // A query with "iteration" but no other markers.
    let no_pronoun = "What are the iteration and batching performance characteristics in database query optimisation?";
    // > 50 chars, no pronoun at word boundary.
    // "in" is not in pronoun list.
    // Verify "it" doesn't match inside "iteration".
    assert!(!FollowUpDetector::is_follow_up(no_pronoun));
}

// ── FollowUpDetector::extract_references ─────────────────────────────────────

#[test]
fn test_extract_references_empty_history() {
    let h = ConversationHistory::new();
    let refs = FollowUpDetector::extract_references(&h);
    assert!(refs.is_empty());
}

#[test]
fn test_extract_references_capitalised_words() {
    let mut h = ConversationHistory::new();
    h.add_turn(Turn::new(TurnRole::User, "Tell me about Rust."));
    h.add_turn(Turn::new(
        TurnRole::Assistant,
        "Rust is a systems programming language created by Mozilla Research.",
    ));
    let refs = FollowUpDetector::extract_references(&h);
    // "Mozilla" and "Research" are capitalised after non-period words.
    assert!(
        refs.iter().any(|r| r == "Mozilla" || r == "Research"),
        "Expected Mozilla or Research in refs, got: {refs:?}"
    );
}

// ── FollowUpDetector::resolve_pronouns ────────────────────────────────────────

#[test]
fn test_resolve_pronouns_no_refs() {
    let result = FollowUpDetector::resolve_pronouns("It is great.", &[]);
    assert_eq!(result, "It is great.");
}

#[test]
fn test_resolve_pronouns_with_refs() {
    let refs = vec!["Rust".to_string(), "Mozilla".to_string()];
    let result = FollowUpDetector::resolve_pronouns("It is fast.", &refs);
    assert!(result.starts_with("Regarding Rust:"));
    assert!(result.contains("It is fast."));
}

#[test]
fn test_resolve_pronouns_no_leading_pronoun() {
    let refs = vec!["Rust".to_string()];
    // "The language" does not start with a pronoun.
    let result = FollowUpDetector::resolve_pronouns("The language is fast.", &refs);
    assert_eq!(result, "The language is fast.");
}

// ── QueryReformulator ─────────────────────────────────────────────────────────

#[test]
fn test_reformulator_empty_query() {
    let r = QueryReformulator::new(ReformulationStrategy::Standalone);
    assert!(matches!(
        r.reformulate("  ", "context"),
        Err(ConversationError::EmptyQuery)
    ));
}

#[test]
fn test_reformulator_standalone() {
    let r = QueryReformulator::new(ReformulationStrategy::Standalone);
    let result = r.reformulate("What is Rust?", "some context").unwrap();
    assert_eq!(result, "What is Rust?");
}

#[test]
fn test_reformulator_concatenation_with_context() {
    let r = QueryReformulator::new(ReformulationStrategy::Concatenation);
    let result = r
        .reformulate("What is its purpose?", "Rust is a language.")
        .unwrap();
    assert!(result.starts_with("Rust is a language."));
    assert!(result.contains("Follow-up: What is its purpose?"));
}

#[test]
fn test_reformulator_concatenation_no_context() {
    let r = QueryReformulator::new(ReformulationStrategy::Concatenation);
    let result = r.reformulate("What is Rust?", "").unwrap();
    assert_eq!(result, "What is Rust?");
}

#[test]
fn test_reformulator_context_injection_with_assistant() {
    let ctx = "User: What is Rust?\nAssistant: Rust is a systems language.";
    let r = QueryReformulator::new(ReformulationStrategy::ContextInjection);
    let result = r.reformulate("What is its memory model?", ctx).unwrap();
    assert!(result.contains("[Context:"));
    assert!(result.contains("Rust is a systems language."));
}

#[test]
fn test_reformulator_context_injection_no_assistant() {
    let ctx = "User: What is Rust?";
    let r = QueryReformulator::new(ReformulationStrategy::ContextInjection);
    let result = r.reformulate("Follow up question here", ctx).unwrap();
    // No assistant turn → query returned unchanged.
    assert_eq!(result, "Follow up question here");
}

#[test]
fn test_reformulator_follow_up_resolution_with_pronoun() {
    let ctx = "User: What is Rust?\nAssistant: The Mozilla Research project.";
    let r = QueryReformulator::new(ReformulationStrategy::FollowUpResolution);
    let result = r.reformulate("It is really interesting!", ctx).unwrap();
    // Should resolve "It" to a reference.
    assert!(
        result.contains("Regarding"),
        "Expected pronoun resolution, got: {result}"
    );
}

#[test]
fn test_reformulator_follow_up_resolution_no_pronoun() {
    let ctx = "User: What is Rust?\nAssistant: A systems language.";
    let r = QueryReformulator::new(ReformulationStrategy::FollowUpResolution);
    let result = r
        .reformulate("Tell me about memory safety in Rust in general.", ctx)
        .unwrap();
    // No leading pronoun, no "Regarding" prefix expected from resolve_pronouns.
    // resolve_pronouns only modifies if starts with pronoun.
    assert!(!result.starts_with("Regarding"));
}

// ── ConversationAwareQuery ─────────────────────────────────────────────────────

#[test]
fn test_conversation_aware_query() {
    let q = ConversationAwareQuery::new("original", "reformulated", "context", true);
    assert_eq!(q.original_query, "original");
    assert_eq!(q.reformulated_query, "reformulated");
    assert_eq!(q.as_search_text(), "reformulated");
    assert!(q.is_follow_up);
}

// ── InMemorySessionManager ────────────────────────────────────────────────────

#[tokio::test]
async fn test_session_manager_create_and_get() {
    let mgr = InMemorySessionManager::new_default();
    let id = mgr.create_session(SessionConfig::default()).await.unwrap();
    let session = mgr.get_session(&id).await.unwrap();
    assert_eq!(session.id, id);
    assert!(session.history.is_empty());
}

#[tokio::test]
async fn test_session_manager_get_not_found() {
    let mgr = InMemorySessionManager::new_default();
    let fake_id = ConversationId::new();
    let result = mgr.get_session(&fake_id).await;
    assert!(matches!(result, Err(ConversationError::SessionNotFound(_))));
}

#[tokio::test]
async fn test_session_manager_update_session() {
    let mgr = InMemorySessionManager::new_default();
    let id = mgr.create_session(SessionConfig::default()).await.unwrap();
    mgr.update_session(&id, Turn::new(TurnRole::User, "hello"))
        .await
        .unwrap();
    let session = mgr.get_session(&id).await.unwrap();
    assert_eq!(session.turn_count(), 1);
    assert_eq!(session.history.last_user_turn().unwrap().text, "hello");
}

#[tokio::test]
async fn test_session_manager_update_not_found() {
    let mgr = InMemorySessionManager::new_default();
    let fake_id = ConversationId::new();
    let result = mgr
        .update_session(&fake_id, Turn::new(TurnRole::User, "q"))
        .await;
    assert!(matches!(result, Err(ConversationError::SessionNotFound(_))));
}

#[tokio::test]
async fn test_session_manager_delete() {
    let mgr = InMemorySessionManager::new_default();
    let id = mgr.create_session(SessionConfig::default()).await.unwrap();
    mgr.delete_session(&id).await.unwrap();
    assert_eq!(mgr.session_count().await, 0);
}

#[tokio::test]
async fn test_session_manager_delete_not_found() {
    let mgr = InMemorySessionManager::new_default();
    let fake_id = ConversationId::new();
    let result = mgr.delete_session(&fake_id).await;
    assert!(matches!(result, Err(ConversationError::SessionNotFound(_))));
}

#[tokio::test]
async fn test_session_manager_list_sessions() {
    let mgr = InMemorySessionManager::new_default();
    let id1 = mgr.create_session(SessionConfig::default()).await.unwrap();
    let id2 = mgr.create_session(SessionConfig::default()).await.unwrap();
    let sessions = mgr.list_sessions().await;
    assert_eq!(sessions.len(), 2);
    assert!(sessions.contains(&id1));
    assert!(sessions.contains(&id2));
}

#[tokio::test]
async fn test_session_manager_session_count() {
    let mgr = InMemorySessionManager::new_default();
    assert_eq!(mgr.session_count().await, 0);
    mgr.create_session(SessionConfig::default()).await.unwrap();
    assert_eq!(mgr.session_count().await, 1);
}

#[tokio::test]
async fn test_session_manager_eviction_on_max_sessions() {
    // max 2 sessions → 3rd create evicts the oldest.
    let mgr = InMemorySessionManager::new(2);
    let _id1 = mgr.create_session(SessionConfig::default()).await.unwrap();
    // Small sleep to ensure created_at ordering is distinct.
    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    let _id2 = mgr.create_session(SessionConfig::default()).await.unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    let _id3 = mgr.create_session(SessionConfig::default()).await.unwrap();
    // Should still have exactly 2 sessions (one was evicted).
    assert_eq!(mgr.session_count().await, 2);
}

#[tokio::test]
async fn test_session_manager_concurrent_access() {
    let mgr = Arc::new(InMemorySessionManager::new_default());
    let id = mgr.create_session(SessionConfig::default()).await.unwrap();

    let mut handles = Vec::new();
    for i in 0..10 {
        let mgr_clone = Arc::clone(&mgr);
        let id_clone = id.clone();
        let handle = tokio::spawn(async move {
            mgr_clone
                .update_session(&id_clone, Turn::new(TurnRole::User, format!("q{i}")))
                .await
                .expect("concurrent update failed");
        });
        handles.push(handle);
    }
    for h in handles {
        h.await.expect("task panicked");
    }

    let session = mgr.get_session(&id).await.unwrap();
    assert_eq!(session.turn_count(), 10);
}

// ── ConversationalPipeline ────────────────────────────────────────────────────

fn make_pipeline() -> ConversationalPipeline<InMemorySessionManager, SlidingWindowBuffer> {
    let mgr = Arc::new(InMemorySessionManager::new_default());
    let buf = SlidingWindowBuffer::new(6);
    let reform = QueryReformulator::new(ReformulationStrategy::FollowUpResolution);
    ConversationalPipeline::new(mgr, buf, reform)
}

#[tokio::test]
async fn test_pipeline_create_and_process() {
    let pipeline = make_pipeline();
    let id = pipeline
        .create_session(SessionConfig::default())
        .await
        .unwrap();
    let aware = pipeline.process_turn(&id, "What is Rust?").await.unwrap();
    assert_eq!(aware.original_query, "What is Rust?");
    assert!(!aware.reformulated_query.is_empty());
}

#[tokio::test]
async fn test_pipeline_empty_query_error() {
    let pipeline = make_pipeline();
    let id = pipeline
        .create_session(SessionConfig::default())
        .await
        .unwrap();
    let result = pipeline.process_turn(&id, "   ").await;
    assert!(matches!(result, Err(ConversationError::EmptyQuery)));
}

#[tokio::test]
async fn test_pipeline_session_not_found_error() {
    let pipeline = make_pipeline();
    let fake_id = ConversationId::new();
    let result = pipeline.process_turn(&fake_id, "Hello").await;
    assert!(matches!(result, Err(ConversationError::SessionNotFound(_))));
}

#[tokio::test]
async fn test_pipeline_record_response() {
    let pipeline = make_pipeline();
    let id = pipeline
        .create_session(SessionConfig::default())
        .await
        .unwrap();
    pipeline
        .record_response(&id, "Rust is a systems language.")
        .await
        .unwrap();
    let session = pipeline.session_manager.get_session(&id).await.unwrap();
    assert_eq!(session.turn_count(), 1);
    assert!(session.history.last_assistant_turn().is_some());
}

#[tokio::test]
async fn test_pipeline_full_turn_cycle() {
    let pipeline = make_pipeline();
    let id = pipeline
        .create_session(SessionConfig::default())
        .await
        .unwrap();

    // Turn 1.
    let _aware1 = pipeline.process_turn(&id, "What is Rust?").await.unwrap();
    pipeline
        .record_user_turn(&id, "What is Rust?")
        .await
        .unwrap();
    pipeline
        .record_response(
            &id,
            "Rust is a systems language focused on safety and speed.",
        )
        .await
        .unwrap();

    // Turn 2 — follow-up.
    let aware2 = pipeline.process_turn(&id, "Tell me more.").await.unwrap();
    assert!(
        aware2.is_follow_up,
        "Short query should be detected as follow-up"
    );
}

#[tokio::test]
async fn test_pipeline_delete_session() {
    let pipeline = make_pipeline();
    let id = pipeline
        .create_session(SessionConfig::default())
        .await
        .unwrap();
    assert_eq!(pipeline.session_count().await, 1);
    pipeline.delete_session(&id).await.unwrap();
    assert_eq!(pipeline.session_count().await, 0);
}

#[tokio::test]
async fn test_pipeline_list_sessions() {
    let pipeline = make_pipeline();
    let id1 = pipeline
        .create_session(SessionConfig::default())
        .await
        .unwrap();
    let id2 = pipeline
        .create_session(SessionConfig::default())
        .await
        .unwrap();
    let sessions = pipeline.list_sessions().await;
    assert!(sessions.contains(&id1));
    assert!(sessions.contains(&id2));
}

#[tokio::test]
async fn test_pipeline_multiple_turns_context_grows() {
    let mgr = Arc::new(InMemorySessionManager::new_default());
    let buf = FullHistoryBuffer;
    let reform = QueryReformulator::new(ReformulationStrategy::Concatenation);
    let pipeline = ConversationalPipeline::new(mgr, buf, reform);

    let id = pipeline
        .create_session(SessionConfig::default())
        .await
        .unwrap();
    pipeline
        .record_user_turn(&id, "What is Rust?")
        .await
        .unwrap();
    pipeline
        .record_response(&id, "A systems language.")
        .await
        .unwrap();

    let aware = pipeline
        .process_turn(&id, "What is its memory model?")
        .await
        .unwrap();

    // Context injection mode: context should contain the history.
    assert!(
        aware.context_used.contains("A systems language."),
        "Context should include prior assistant turn"
    );
}

#[tokio::test]
async fn test_pipeline_with_summary_buffer() {
    let mgr = Arc::new(InMemorySessionManager::new_default());
    let buf = SummaryBuffer::new(4, 500);
    let reform = QueryReformulator::new(ReformulationStrategy::FollowUpResolution);
    let pipeline = ConversationalPipeline::new(mgr, buf, reform);

    let id = pipeline
        .create_session(SessionConfig::default())
        .await
        .unwrap();
    for i in 0..15 {
        pipeline
            .record_user_turn(&id, &format!("question {i}"))
            .await
            .unwrap();
        pipeline
            .record_response(&id, &format!("answer {i}"))
            .await
            .unwrap();
    }

    let aware = pipeline
        .process_turn(&id, "What about the latest developments?")
        .await
        .unwrap();
    assert!(!aware.reformulated_query.is_empty());
}

// ── Serde round-trip ──────────────────────────────────────────────────────────

#[test]
fn test_turn_serde_round_trip() {
    let t = Turn::new(TurnRole::User, "Hello world").with_metadata("k", "v");
    let json = serde_json::to_string(&t).expect("serialise Turn");
    let t2: Turn = serde_json::from_str(&json).expect("deserialise Turn");
    assert_eq!(t2.text, t.text);
    assert_eq!(t2.role, t.role);
    assert_eq!(t2.metadata.get("k").map(String::as_str), Some("v"));
}

#[test]
fn test_session_config_serde_round_trip() {
    let cfg = SessionConfig::default().with_max_turns(42);
    let json = serde_json::to_string(&cfg).expect("serialise SessionConfig");
    let cfg2: SessionConfig = serde_json::from_str(&json).expect("deserialise SessionConfig");
    assert_eq!(cfg2.max_turns, 42);
}

#[test]
fn test_conversation_id_serde_round_trip() {
    let id = ConversationId::new();
    let json = serde_json::to_string(&id).expect("serialise ConversationId");
    let id2: ConversationId = serde_json::from_str(&json).expect("deserialise ConversationId");
    assert_eq!(id, id2);
}
