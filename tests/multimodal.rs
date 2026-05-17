//! Compile-time trait bound checks for multi-modal embeddings.
//!
//! These tests do not download any models. They simply verify that
//! the type system wires up correctly.
#[cfg(feature = "multimodal")]
mod tests {
    use oxirag::layer1_echo::EmbeddingInput;

    #[test]
    fn test_embedding_input_variants_compile() {
        let _text = EmbeddingInput::Text("hello");
        let _img = EmbeddingInput::Image(&[0u8, 1, 2]);
        let _both = EmbeddingInput::TextAndImage {
            text: "caption",
            image: &[0u8, 1],
        };
    }

    #[test]
    fn test_embedding_input_text_clone_roundtrip() {
        let original = EmbeddingInput::Text("roundtrip text");
        let cloned = original.clone();
        // Both should be the Text variant wrapping the same str content.
        if let (EmbeddingInput::Text(a), EmbeddingInput::Text(b)) = (&original, &cloned) {
            assert_eq!(a, b);
        } else {
            panic!("clone changed variant");
        }
    }

    #[test]
    fn test_embedding_input_image_clone_roundtrip() {
        let data = [10u8, 20, 30];
        let original = EmbeddingInput::Image(&data);
        let cloned = original.clone();
        if let (EmbeddingInput::Image(a), EmbeddingInput::Image(b)) = (&original, &cloned) {
            assert_eq!(a, b);
        } else {
            panic!("clone changed variant");
        }
    }

    #[test]
    fn test_embedding_input_joint_clone_roundtrip() {
        let data = [5u8, 6, 7];
        let original = EmbeddingInput::TextAndImage {
            text: "a caption",
            image: &data,
        };
        let cloned = original.clone();
        if let (
            EmbeddingInput::TextAndImage {
                text: ta,
                image: ia,
            },
            EmbeddingInput::TextAndImage {
                text: tb,
                image: ib,
            },
        ) = (&original, &cloned)
        {
            assert_eq!(ta, tb);
            assert_eq!(ia, ib);
        } else {
            panic!("clone changed variant");
        }
    }

    #[test]
    fn test_embedding_input_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        // EmbeddingInput is a borrowed-data type; this confirms the lifetime
        // doesn't break Send+Sync for the static lifetime specialisation.
        assert_send_sync::<EmbeddingInput<'static>>();
    }
}
