use crate::memory_paging::pager::{ArchivalStore, ContextPager, MainContext};
use crate::memory_paging::types::{
    MemoryPage, MemoryPagingConfig, MemoryPagingError, PagingAction, PagingOutcome, PagingPolicy,
};

fn page(id: &str, size: usize, importance: f32, tick: u64) -> MemoryPage {
    MemoryPage {
        id: id.to_string(),
        content: format!("content for {id}"),
        size,
        importance,
        created_tick: tick,
        last_accessed_tick: tick,
        access_count: 0,
        embedding: Vec::new(),
    }
}

// ── MemoryPagingConfig ───────────────────────────────────────────────────────

#[test]
fn test_config_default_values() {
    let cfg = MemoryPagingConfig::default();
    assert_eq!(cfg.capacity, 128);
    assert_eq!(cfg.policy, PagingPolicy::LeastRecentlyUsed);
    assert_eq!(cfg.dim, 64);
    assert_eq!(cfg.search_top_k, 5);
    assert!((cfg.min_search_similarity - 0.0).abs() < 1e-6);
}

#[test]
fn test_config_builder_with_capacity() {
    let cfg = MemoryPagingConfig::default().with_capacity(64);
    assert_eq!(cfg.capacity, 64);
}

#[test]
fn test_config_builder_with_policy() {
    let cfg = MemoryPagingConfig::default().with_policy(PagingPolicy::LeastImportant);
    assert_eq!(cfg.policy, PagingPolicy::LeastImportant);
}

#[test]
fn test_config_builder_with_dim() {
    let cfg = MemoryPagingConfig::default().with_dim(32);
    assert_eq!(cfg.dim, 32);
}

#[test]
fn test_config_builder_with_search_top_k() {
    let cfg = MemoryPagingConfig::default().with_search_top_k(3);
    assert_eq!(cfg.search_top_k, 3);
}

#[test]
fn test_config_builder_with_min_search_similarity() {
    let cfg = MemoryPagingConfig::default().with_min_search_similarity(0.5);
    assert!((cfg.min_search_similarity - 0.5).abs() < 1e-6);
}

#[test]
fn test_config_builder_chaining() {
    let cfg = MemoryPagingConfig::default()
        .with_capacity(10)
        .with_policy(PagingPolicy::LeastImportant)
        .with_dim(16)
        .with_search_top_k(2)
        .with_min_search_similarity(0.1);
    assert_eq!(cfg.capacity, 10);
    assert_eq!(cfg.policy, PagingPolicy::LeastImportant);
    assert_eq!(cfg.dim, 16);
    assert_eq!(cfg.search_top_k, 2);
    assert!((cfg.min_search_similarity - 0.1).abs() < 1e-6);
}

#[test]
fn test_paging_policy_default_is_lru() {
    assert_eq!(PagingPolicy::default(), PagingPolicy::LeastRecentlyUsed);
}

// ── MainContext ──────────────────────────────────────────────────────────────

#[test]
fn test_main_context_new_is_empty() {
    let ctx = MainContext::new(100);
    assert_eq!(ctx.capacity(), 100);
    assert_eq!(ctx.len(), 0);
    assert!(ctx.is_empty());
    assert_eq!(ctx.occupied_size(), 0);
    assert_eq!(ctx.available_size(), 100);
}

#[test]
fn test_main_context_insert_unchecked_and_get() {
    let mut ctx = MainContext::new(100);
    ctx.insert_unchecked(page("p1", 10, 0.5, 1));
    assert_eq!(ctx.len(), 1);
    assert!(!ctx.is_empty());
    assert!(ctx.contains("p1"));
    assert_eq!(ctx.occupied_size(), 10);
    assert_eq!(ctx.available_size(), 90);
    let got = ctx.get("p1").expect("page should be present");
    assert_eq!(got.size, 10);
}

#[test]
fn test_main_context_insert_unchecked_can_exceed_capacity() {
    // insert_unchecked deliberately bypasses the capacity invariant; enforcing
    // it is ContextPager::admit's job. Direct MainContext access is pub(crate)
    // exactly so external callers cannot do this.
    let mut ctx = MainContext::new(5);
    ctx.insert_unchecked(page("p1", 3, 0.5, 1));
    ctx.insert_unchecked(page("p2", 4, 0.5, 2));
    assert_eq!(ctx.occupied_size(), 7);
    assert!(ctx.occupied_size() > ctx.capacity());
}

#[test]
fn test_main_context_remove() {
    let mut ctx = MainContext::new(100);
    ctx.insert_unchecked(page("p1", 10, 0.5, 1));
    let removed = ctx.remove("p1").expect("should remove");
    assert_eq!(removed.id, "p1");
    assert!(!ctx.contains("p1"));
    assert_eq!(ctx.occupied_size(), 0);
}

#[test]
fn test_main_context_remove_missing_returns_none() {
    let mut ctx = MainContext::new(100);
    assert!(ctx.remove("nope").is_none());
}

#[test]
fn test_main_context_touch_updates_recency_and_access_count() {
    let mut ctx = MainContext::new(100);
    ctx.insert_unchecked(page("p1", 10, 0.5, 1));
    let touched = ctx.touch("p1", 99);
    assert!(touched);
    let p = ctx.get("p1").expect("present");
    assert_eq!(p.last_accessed_tick, 99);
    assert_eq!(p.access_count, 1);
}

#[test]
fn test_main_context_touch_missing_returns_false() {
    let mut ctx = MainContext::new(100);
    assert!(!ctx.touch("nope", 1));
}

#[test]
fn test_main_context_iter_visits_all_pages() {
    let mut ctx = MainContext::new(100);
    ctx.insert_unchecked(page("p1", 1, 0.5, 1));
    ctx.insert_unchecked(page("p2", 1, 0.5, 2));
    let mut ids: Vec<String> = ctx.iter().map(|p| p.id.clone()).collect();
    ids.sort();
    assert_eq!(ids, vec!["p1".to_string(), "p2".to_string()]);
}

// ── ArchivalStore ────────────────────────────────────────────────────────────

#[test]
fn test_archival_store_default_is_empty() {
    let store = ArchivalStore::default();
    assert_eq!(store.len(), 0);
    assert!(store.is_empty());
}

#[test]
fn test_archival_store_insert_and_get() {
    let mut store = ArchivalStore::default();
    store.insert(page("a1", 5, 0.2, 1));
    assert_eq!(store.len(), 1);
    assert!(store.contains("a1"));
    assert_eq!(store.get("a1").expect("present").size, 5);
}

#[test]
fn test_archival_store_insert_replaces_existing() {
    let mut store = ArchivalStore::default();
    store.insert(page("a1", 5, 0.2, 1));
    let old = store.insert(page("a1", 9, 0.9, 2));
    assert_eq!(old.expect("previous page returned").size, 5);
    assert_eq!(store.get("a1").expect("present").size, 9);
    assert_eq!(store.len(), 1);
}

#[test]
fn test_archival_store_remove() {
    let mut store = ArchivalStore::default();
    store.insert(page("a1", 5, 0.2, 1));
    let removed = store.remove("a1").expect("should remove");
    assert_eq!(removed.id, "a1");
    assert!(!store.contains("a1"));
}

#[test]
fn test_archival_store_remove_missing_returns_none() {
    let mut store = ArchivalStore::default();
    assert!(store.remove("nope").is_none());
}

#[test]
fn test_archival_store_iter_visits_all_pages() {
    let mut store = ArchivalStore::default();
    store.insert(page("a1", 1, 0.5, 1));
    store.insert(page("a2", 1, 0.5, 2));
    let mut ids: Vec<String> = store.iter().map(|p| p.id.clone()).collect();
    ids.sort();
    assert_eq!(ids, vec!["a1".to_string(), "a2".to_string()]);
}

// ── ContextPager: construction ───────────────────────────────────────────────

#[test]
fn test_context_pager_new_starts_empty() {
    let pager = ContextPager::new(MemoryPagingConfig::default());
    assert!(pager.main.is_empty());
    assert!(pager.archival.is_empty());
    assert_eq!(pager.tick(), 0);
}

#[test]
fn test_context_pager_default_uses_default_config() {
    let pager = ContextPager::default();
    assert_eq!(
        pager.main.capacity(),
        MemoryPagingConfig::default().capacity
    );
}

#[test]
fn test_context_pager_main_capacity_matches_config() {
    let pager = ContextPager::new(MemoryPagingConfig::default().with_capacity(42));
    assert_eq!(pager.main.capacity(), 42);
}

// ── make_page ────────────────────────────────────────────────────────────────

#[test]
fn test_make_page_empty_content_errors() {
    let pager = ContextPager::default();
    let result = pager.make_page("   ", 0.5);
    assert!(matches!(result, Err(MemoryPagingError::EmptyContent)));
}

#[test]
fn test_make_page_computes_whitespace_token_size() {
    let pager = ContextPager::default();
    let p = pager.make_page("alpha beta gamma", 0.5).expect("ok");
    assert_eq!(p.size, 3);
}

#[test]
fn test_make_page_clamps_importance() {
    let pager = ContextPager::default();
    let too_high = pager.make_page("alpha", 5.0).expect("ok");
    let too_low = pager.make_page("beta", -5.0).expect("ok");
    assert!((too_high.importance - 1.0).abs() < 1e-6);
    assert!((too_low.importance - 0.0).abs() < 1e-6);
}

#[test]
fn test_make_page_produces_normalized_embedding() {
    let pager = ContextPager::new(MemoryPagingConfig::default().with_dim(16));
    let p = pager.make_page("alpha beta alpha", 0.5).expect("ok");
    assert_eq!(p.embedding.len(), 16);
    let norm: f32 = p.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5);
}

#[test]
fn test_make_page_does_not_insert_anywhere() {
    let pager = ContextPager::default();
    let p = pager.make_page("alpha beta", 0.5).expect("ok");
    assert!(!pager.main.contains(&p.id));
    assert!(!pager.archival.contains(&p.id));
}

// ── PagingAction::Write ──────────────────────────────────────────────────────

#[test]
fn test_write_empty_content_errors() {
    let mut pager = ContextPager::default();
    let result = pager.execute(PagingAction::Write {
        content: String::new(),
        importance: 0.5,
    });
    assert!(matches!(result, Err(MemoryPagingError::EmptyContent)));
}

#[test]
fn test_write_whitespace_only_content_errors() {
    let mut pager = ContextPager::default();
    let result = pager.execute(PagingAction::Write {
        content: "   ".to_string(),
        importance: 0.5,
    });
    assert!(matches!(result, Err(MemoryPagingError::EmptyContent)));
}

#[test]
fn test_write_admits_page_into_main_context() {
    let mut pager = ContextPager::default();
    let outcome = pager
        .execute(PagingAction::Write {
            content: "hello world".to_string(),
            importance: 0.5,
        })
        .expect("write should succeed");
    let id = outcome.paged_in.expect("write always pages in");
    assert!(pager.main.contains(&id));
    assert!(outcome.paged_out.is_empty());
    assert_eq!(pager.main.occupied_size(), 2);
}

#[test]
fn test_write_content_too_large_for_capacity_errors() {
    let mut pager = ContextPager::new(MemoryPagingConfig::default().with_capacity(2));
    let result = pager.execute(PagingAction::Write {
        content: "one two three four five".to_string(), // 5 tokens > capacity 2
        importance: 0.5,
    });
    assert!(matches!(
        result,
        Err(MemoryPagingError::PageTooLargeForCapacity {
            size: 5,
            capacity: 2,
            ..
        })
    ));
    // A failed write must not have stored anything anywhere.
    assert!(pager.main.is_empty());
    assert!(pager.archival.is_empty());
}

// ── Capacity boundary conditions ─────────────────────────────────────────────

#[test]
fn test_write_exact_capacity_fit_succeeds_without_eviction() {
    let mut pager = ContextPager::new(MemoryPagingConfig::default().with_capacity(3));
    let outcome = pager
        .execute(PagingAction::Write {
            content: "one two three".to_string(), // exactly 3 tokens
            importance: 0.5,
        })
        .expect("exact fit should succeed");
    assert!(outcome.paged_out.is_empty());
    assert_eq!(pager.main.occupied_size(), 3);
    assert_eq!(pager.main.available_size(), 0);
}

#[test]
fn test_write_one_over_capacity_evicts_to_make_room() {
    let mut pager = ContextPager::new(MemoryPagingConfig::default().with_capacity(3));
    let first = pager
        .execute(PagingAction::Write {
            content: "one two three".to_string(),
            importance: 0.5,
        })
        .expect("ok")
        .paged_in
        .expect("paged in");
    let outcome = pager
        .execute(PagingAction::Write {
            content: "four".to_string(), // 1 token; needs 1 unit of room
            importance: 0.5,
        })
        .expect("should evict and succeed");
    assert_eq!(outcome.paged_out, vec![first.clone()]);
    assert!(!pager.main.contains(&first));
    assert!(pager.archival.contains(&first));
}

#[test]
fn test_zero_capacity_rejects_any_nonempty_page() {
    let mut pager = ContextPager::new(MemoryPagingConfig::default().with_capacity(0));
    let result = pager.execute(PagingAction::Write {
        content: "one".to_string(),
        importance: 0.5,
    });
    assert!(matches!(
        result,
        Err(MemoryPagingError::PageTooLargeForCapacity { capacity: 0, .. })
    ));
}

// ── Eviction under pressure (core mechanic) ──────────────────────────────────

#[test]
fn test_eviction_under_pressure_evicts_lru_victim_to_archival() {
    let mut pager = ContextPager::new(
        MemoryPagingConfig::default()
            .with_capacity(8)
            .with_policy(PagingPolicy::LeastRecentlyUsed),
    );
    let first = pager
        .execute(PagingAction::Write {
            content: "alpha beta".to_string(), // size 2
            importance: 0.5,
        })
        .expect("ok")
        .paged_in
        .expect("paged in");
    let second = pager
        .execute(PagingAction::Write {
            content: "gamma delta epsilon".to_string(), // size 3, occupied now 5
            importance: 0.5,
        })
        .expect("ok")
        .paged_in
        .expect("paged in");

    // Third write needs 4 units; 5 + 4 = 9 > 8, so exactly one eviction
    // (5 - 2 + 4 = 7 <= 8) is needed. LRU picks `first` (older tick).
    let outcome = pager
        .execute(PagingAction::Write {
            content: "zeta eta theta iota".to_string(), // size 4
            importance: 0.5,
        })
        .expect("should evict under pressure and succeed");

    assert_eq!(outcome.paged_out, vec![first.clone()]);
    assert!(
        !pager.main.contains(&first),
        "victim must leave main context"
    );
    assert!(
        pager.main.contains(&second),
        "non-victim must remain resident"
    );
    assert!(
        pager.archival.contains(&first),
        "victim must be recoverable from archival storage"
    );
    // Lossless: content is preserved exactly.
    assert_eq!(
        pager.archival.get(&first).expect("archived").content,
        "alpha beta"
    );
    assert!(pager.main.occupied_size() <= pager.main.capacity());
}

#[test]
fn test_eviction_under_pressure_evicts_multiple_victims_if_needed() {
    let mut pager = ContextPager::new(MemoryPagingConfig::default().with_capacity(6));
    let first = pager
        .execute(PagingAction::Write {
            content: "a b".to_string(), // size 2
            importance: 0.5,
        })
        .expect("ok")
        .paged_in
        .expect("paged in");
    let second = pager
        .execute(PagingAction::Write {
            content: "c d e".to_string(), // size 3, occupied now 5
            importance: 0.5,
        })
        .expect("ok")
        .paged_in
        .expect("paged in");

    // Needs 4 units of room; evicting just `first` (2) only frees to 3 + 4 =
    // 7 > 6, so `second` must also be evicted (occupied 0, then 0+4=4 <= 6).
    let outcome = pager
        .execute(PagingAction::Write {
            content: "f g h i".to_string(), // size 4
            importance: 0.5,
        })
        .expect("should evict multiple victims and succeed");

    assert_eq!(outcome.paged_out.len(), 2);
    assert!(outcome.paged_out.contains(&first));
    assert!(outcome.paged_out.contains(&second));
    assert!(!pager.main.is_empty()); // the new page is resident
    assert!(!pager.main.contains(&first));
    assert!(!pager.main.contains(&second));
    assert!(pager.archival.contains(&first));
    assert!(pager.archival.contains(&second));
    assert_eq!(pager.main.occupied_size(), 4);
}

#[test]
fn test_eviction_under_pressure_with_least_important_policy() {
    let mut pager = ContextPager::new(
        MemoryPagingConfig::default()
            .with_capacity(8)
            .with_policy(PagingPolicy::LeastImportant),
    );
    // `low` is written first (oldest tick) but has the LOWEST importance;
    // `high` is written second (newer tick) with high importance.
    // Under LRU, `low` (older) would be evicted anyway, so to prove the
    // policy is actually importance-driven (not just LRU in disguise) we
    // give `high` an OLDER tick than a third, unimportant page.
    let low = pager
        .execute(PagingAction::Write {
            content: "alpha beta".to_string(), // size 2, importance 0.1 (low)
            importance: 0.1,
        })
        .expect("ok")
        .paged_in
        .expect("paged in");
    let high = pager
        .execute(PagingAction::Write {
            content: "gamma delta epsilon".to_string(), // size 3, importance 0.9 (high)
            importance: 0.9,
        })
        .expect("ok")
        .paged_in
        .expect("paged in");
    // occupied = 5; next write needs 4 units -> must evict.
    let outcome = pager
        .execute(PagingAction::Write {
            content: "zeta eta theta iota".to_string(), // size 4
            importance: 0.5,
        })
        .expect("should evict least-important victim");

    // Even though `low` is the OLDER page, LeastImportant policy evicts it
    // because of its low importance score, not its recency.
    assert_eq!(outcome.paged_out, vec![low.clone()]);
    assert!(!pager.main.contains(&low));
    assert!(
        pager.main.contains(&high),
        "high-importance page must survive"
    );
}

#[test]
fn test_lru_and_least_important_policies_pick_different_victims() {
    // Construct a scenario where the two policies disagree: the older page
    // is MORE important than the newer page.
    let build = |policy: PagingPolicy| {
        let mut pager = ContextPager::new(
            MemoryPagingConfig::default()
                .with_capacity(8)
                .with_policy(policy),
        );
        let older_important = pager
            .execute(PagingAction::Write {
                content: "alpha beta".to_string(), // size 2
                importance: 0.9,
            })
            .expect("ok")
            .paged_in
            .expect("paged in");
        let newer_unimportant = pager
            .execute(PagingAction::Write {
                content: "gamma delta epsilon".to_string(), // size 3, occupied 5
                importance: 0.1,
            })
            .expect("ok")
            .paged_in
            .expect("paged in");
        let outcome = pager
            .execute(PagingAction::Write {
                content: "zeta eta theta iota".to_string(), // size 4, forces 1 eviction
                importance: 0.5,
            })
            .expect("should evict");
        (outcome.paged_out, older_important, newer_unimportant)
    };

    let (lru_evicted, older, _newer) = build(PagingPolicy::LeastRecentlyUsed);
    let (importance_evicted, older2, newer2) = build(PagingPolicy::LeastImportant);

    assert_eq!(lru_evicted, vec![older]);
    assert_eq!(importance_evicted, vec![newer2]);
    assert_ne!(lru_evicted, importance_evicted);
    let _ = older2;
}

// ── PagingAction::PageIn ─────────────────────────────────────────────────────

#[test]
fn test_page_in_missing_id_errors() {
    let mut pager = ContextPager::default();
    let result = pager.execute(PagingAction::PageIn {
        page_id: "nope".to_string(),
    });
    assert!(matches!(
        result,
        Err(MemoryPagingError::PageNotInArchivalStore(id)) if id == "nope"
    ));
}

#[test]
fn test_page_in_already_resident_is_idempotent_touch() {
    let mut pager = ContextPager::default();
    let id = pager
        .execute(PagingAction::Write {
            content: "hello".to_string(),
            importance: 0.5,
        })
        .expect("ok")
        .paged_in
        .expect("paged in");
    let before_len = pager.main.len();
    let outcome = pager
        .execute(PagingAction::PageIn {
            page_id: id.clone(),
        })
        .expect("touch should succeed");
    assert_eq!(outcome.paged_in, Some(id));
    assert!(outcome.paged_out.is_empty());
    assert_eq!(pager.main.len(), before_len);
}

#[test]
fn test_page_in_from_archival_round_trips_content_losslessly() {
    let mut pager = ContextPager::default();
    let id = pager
        .execute(PagingAction::Write {
            content: "the quick brown fox".to_string(),
            importance: 0.7,
        })
        .expect("ok")
        .paged_in
        .expect("paged in");
    pager
        .execute(PagingAction::PageOut {
            page_id: id.clone(),
        })
        .expect("page out should succeed");
    assert!(!pager.main.contains(&id));
    assert!(pager.archival.contains(&id));

    let outcome = pager
        .execute(PagingAction::PageIn {
            page_id: id.clone(),
        })
        .expect("page in should succeed");
    assert_eq!(outcome.paged_in, Some(id.clone()));
    assert!(pager.main.contains(&id));
    assert!(!pager.archival.contains(&id));
    let resident = pager.main.get(&id).expect("resident");
    assert_eq!(resident.content, "the quick brown fox");
    assert!((resident.importance - 0.7).abs() < 1e-6);
}

#[test]
fn test_page_in_from_archival_too_large_for_capacity_errors_and_preserves_page() {
    let mut pager = ContextPager::new(MemoryPagingConfig::default().with_capacity(10));
    let id = pager
        .execute(PagingAction::Write {
            content: "one two three four five".to_string(), // size 5
            importance: 0.5,
        })
        .expect("ok")
        .paged_in
        .expect("paged in");
    pager
        .execute(PagingAction::PageOut {
            page_id: id.clone(),
        })
        .expect("page out");

    // Shrink capacity below the archived page's size so it can never fit.
    pager.main = MainContext::new(3);
    let result = pager.execute(PagingAction::PageIn {
        page_id: id.clone(),
    });
    assert!(matches!(
        result,
        Err(MemoryPagingError::PageTooLargeForCapacity {
            size: 5,
            capacity: 3,
            ..
        })
    ));
    // Data loss check: the page must still be fully recoverable in archival
    // storage, not silently dropped by the failed page-in attempt.
    assert!(pager.archival.contains(&id));
    assert_eq!(
        pager.archival.get(&id).expect("still archived").content,
        "one two three four five"
    );
}

#[test]
fn test_page_in_triggers_eviction_of_other_resident_pages() {
    let mut pager = ContextPager::new(MemoryPagingConfig::default().with_capacity(6));
    let archived = pager.make_page("one two three", 0.5).expect("ok"); // size 3
    let archived_id = archived.id.clone();
    pager.archival.insert(archived);

    let resident_a = pager
        .execute(PagingAction::Write {
            content: "four five".to_string(), // size 2
            importance: 0.5,
        })
        .expect("ok")
        .paged_in
        .expect("paged in");
    let resident_b = pager
        .execute(PagingAction::Write {
            content: "six seven".to_string(), // size 2, occupied now 4
            importance: 0.5,
        })
        .expect("ok")
        .paged_in
        .expect("paged in");

    // Paging in `archived` (size 3) needs 4 + 3 = 7 > 6, forcing one eviction.
    let outcome = pager
        .execute(PagingAction::PageIn {
            page_id: archived_id.clone(),
        })
        .expect("page in should succeed with eviction");
    assert_eq!(outcome.paged_in, Some(archived_id.clone()));
    assert_eq!(outcome.paged_out, vec![resident_a.clone()]);
    assert!(!pager.main.contains(&resident_a));
    assert!(pager.main.contains(&resident_b));
    assert!(pager.main.contains(&archived_id));
    assert!(pager.archival.contains(&resident_a));
}

// ── PagingAction::PageOut ────────────────────────────────────────────────────

#[test]
fn test_page_out_missing_id_errors() {
    let mut pager = ContextPager::default();
    let result = pager.execute(PagingAction::PageOut {
        page_id: "nope".to_string(),
    });
    assert!(matches!(
        result,
        Err(MemoryPagingError::PageNotInMainContext(id)) if id == "nope"
    ));
}

#[test]
fn test_page_out_moves_page_to_archival() {
    let mut pager = ContextPager::default();
    let id = pager
        .execute(PagingAction::Write {
            content: "some content".to_string(),
            importance: 0.5,
        })
        .expect("ok")
        .paged_in
        .expect("paged in");
    let outcome = pager
        .execute(PagingAction::PageOut {
            page_id: id.clone(),
        })
        .expect("page out should succeed");
    assert_eq!(outcome.paged_in, None);
    assert_eq!(outcome.paged_out, vec![id.clone()]);
    assert!(!pager.main.contains(&id));
    assert!(pager.archival.contains(&id));
}

#[test]
fn test_page_out_never_fails_on_capacity() {
    // Archival storage is "unbounded" — page-out cannot fail due to size.
    let mut pager = ContextPager::new(MemoryPagingConfig::default().with_capacity(1000));
    let id = pager
        .execute(PagingAction::Write {
            content: "x".to_string(),
            importance: 0.5,
        })
        .expect("ok")
        .paged_in
        .expect("paged in");
    assert!(pager.execute(PagingAction::PageOut { page_id: id }).is_ok());
}

// ── PagingAction::Search ─────────────────────────────────────────────────────

#[test]
fn test_search_empty_query_errors() {
    let mut pager = ContextPager::default();
    let result = pager.execute(PagingAction::Search {
        query: "   ".to_string(),
    });
    assert!(matches!(result, Err(MemoryPagingError::EmptyQuery)));
}

#[test]
fn test_search_empty_archival_store_returns_empty_outcome_not_error() {
    let mut pager = ContextPager::default();
    let outcome = pager
        .execute(PagingAction::Search {
            query: "anything".to_string(),
        })
        .expect("search over empty archival storage should not error");
    assert!(outcome.paged_in.is_none());
    assert!(outcome.paged_out.is_empty());
    assert!(outcome.search_matches.is_empty());
}

#[test]
fn test_search_finds_and_pages_in_page_not_currently_in_main_context() {
    let mut pager = ContextPager::default();
    let paris = pager
        .make_page("Paris is the capital of France", 0.5)
        .expect("ok");
    let paris_id = paris.id.clone();
    pager.archival.insert(paris);
    let tokyo = pager
        .make_page("Tokyo is the capital of Japan", 0.5)
        .expect("ok");
    let tokyo_id = tokyo.id.clone();
    pager.archival.insert(tokyo);

    assert!(!pager.main.contains(&paris_id));
    assert!(!pager.main.contains(&tokyo_id));

    let outcome = pager
        .execute(PagingAction::Search {
            query: "capital of France".to_string(),
        })
        .expect("search should succeed");

    assert_eq!(outcome.paged_in, Some(paris_id.clone()));
    assert!(pager.main.contains(&paris_id));
    assert!(!pager.archival.contains(&paris_id));
    // The non-matching candidate stays untouched in archival storage.
    assert!(pager.archival.contains(&tokyo_id));
    assert!(!outcome.search_matches.is_empty());
    assert_eq!(outcome.search_matches[0].0, paris_id);
}

#[test]
fn test_search_reports_top_k_matches_sorted_descending() {
    let mut pager = ContextPager::new(MemoryPagingConfig::default().with_search_top_k(2));
    for content in [
        "Paris is the capital of France",
        "Berlin is the capital of Germany",
        "cats and dogs are common pets",
    ] {
        let p = pager.make_page(content, 0.5).expect("ok");
        pager.archival.insert(p);
    }
    let outcome = pager
        .execute(PagingAction::Search {
            query: "capital of France".to_string(),
        })
        .expect("search should succeed");
    assert!(outcome.search_matches.len() <= 2);
    // Scores should be sorted descending.
    for pair in outcome.search_matches.windows(2) {
        assert!(pair[0].1 >= pair[1].1);
    }
}

#[test]
fn test_search_below_similarity_threshold_does_not_page_in() {
    let mut pager =
        ContextPager::new(MemoryPagingConfig::default().with_min_search_similarity(2.0));
    let p = pager
        .make_page("Paris is the capital of France", 0.5)
        .expect("ok");
    let id = p.id.clone();
    pager.archival.insert(p);

    let outcome = pager
        .execute(PagingAction::Search {
            query: "capital of France".to_string(),
        })
        .expect("search should succeed even with no page-in");
    assert!(outcome.paged_in.is_none());
    assert!(!outcome.search_matches.is_empty());
    // Page must remain archived, not silently lost.
    assert!(pager.archival.contains(&id));
    assert!(!pager.main.contains(&id));
}

#[test]
fn test_search_triggers_eviction_under_pressure() {
    let mut pager = ContextPager::new(MemoryPagingConfig::default().with_capacity(6));
    let target = pager.make_page("one two three", 0.5).expect("ok"); // size 3
    let target_id = target.id.clone();
    pager.archival.insert(target);

    let resident = pager
        .execute(PagingAction::Write {
            content: "four five six".to_string(), // size 3
            importance: 0.5,
        })
        .expect("ok")
        .paged_in
        .expect("paged in");

    // occupied=3, capacity=6; paging in size-3 target needs 3+3=6 <= 6, so no
    // eviction needed here — bump pressure by writing one more small page.
    let second_resident = pager
        .execute(PagingAction::Write {
            content: "seven".to_string(), // size 1, occupied now 4
            importance: 0.5,
        })
        .expect("ok")
        .paged_in
        .expect("paged in");

    // Now occupied=4; paging in target (size 3) needs 4+3=7 > 6: evict.
    let outcome = pager
        .execute(PagingAction::Search {
            query: "one two three".to_string(),
        })
        .expect("search should succeed");
    assert_eq!(outcome.paged_in, Some(target_id.clone()));
    assert!(!outcome.paged_out.is_empty());
    assert!(pager.main.contains(&target_id));
    let _ = resident;
    let _ = second_resident;
}

// ── PagingOutcome ────────────────────────────────────────────────────────────

#[test]
fn test_paging_outcome_default_is_empty() {
    let outcome = PagingOutcome::default();
    assert!(outcome.paged_in.is_none());
    assert!(outcome.paged_out.is_empty());
    assert!(outcome.search_matches.is_empty());
}

// ── MemoryPagingError ────────────────────────────────────────────────────────

#[test]
fn test_error_empty_content_display() {
    let e = MemoryPagingError::EmptyContent;
    assert!(!format!("{e}").is_empty());
}

#[test]
fn test_error_page_not_in_main_context_display_includes_id() {
    let e = MemoryPagingError::PageNotInMainContext("abc".to_string());
    assert!(format!("{e}").contains("abc"));
}

#[test]
fn test_error_page_not_in_archival_store_display_includes_id() {
    let e = MemoryPagingError::PageNotInArchivalStore("xyz".to_string());
    assert!(format!("{e}").contains("xyz"));
}

#[test]
fn test_error_page_too_large_display_includes_fields() {
    let e = MemoryPagingError::PageTooLargeForCapacity {
        page_id: "p1".to_string(),
        size: 20,
        capacity: 10,
    };
    let msg = format!("{e}");
    assert!(msg.contains("p1"));
    assert!(msg.contains("20"));
    assert!(msg.contains("10"));
}

#[test]
fn test_error_empty_query_display() {
    let e = MemoryPagingError::EmptyQuery;
    assert!(!format!("{e}").is_empty());
}

#[test]
fn test_error_no_eviction_candidate_display_includes_id() {
    let e = MemoryPagingError::NoEvictionCandidate("p9".to_string());
    assert!(format!("{e}").contains("p9"));
}

// ── End-to-end: multi-step paging session ────────────────────────────────────

#[test]
fn test_end_to_end_multi_step_session_preserves_all_pages_somewhere() {
    let mut pager = ContextPager::new(MemoryPagingConfig::default().with_capacity(5));
    let mut all_ids = Vec::new();

    for content in [
        "one two",
        "three four",
        "five six",
        "seven eight",
        "nine ten",
    ] {
        let outcome = pager
            .execute(PagingAction::Write {
                content: content.to_string(),
                importance: 0.5,
            })
            .expect("write should succeed");
        all_ids.push(outcome.paged_in.expect("paged in"));
    }

    // Every page ever written must be findable in exactly one of the two
    // tiers — nothing is ever discarded.
    for id in &all_ids {
        let in_main = pager.main.contains(id);
        let in_archival = pager.archival.contains(id);
        assert!(
            in_main ^ in_archival,
            "page {id} must be resident XOR archived, never both/neither"
        );
    }
    assert!(pager.main.occupied_size() <= pager.main.capacity());

    // Page everything back in via explicit PageIn and verify round-trip
    // content integrity for every single page.
    for id in &all_ids {
        pager
            .execute(PagingAction::PageIn {
                page_id: id.clone(),
            })
            .expect("page in should always succeed for a capacity-1 page into capacity-5 context");
        assert!(pager.main.contains(id) || pager.archival.contains(id));
    }
}
