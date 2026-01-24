//! Metadata filtering for vector search.
//!
//! This module provides a flexible metadata filtering system for the Echo layer,
//! allowing documents to be filtered based on their metadata during vector search.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A filter that can be applied to document metadata during vector search.
///
/// `MetadataFilter` supports various comparison operations and can be composed
/// using logical AND and OR operations for complex filtering scenarios.
///
/// # Examples
///
/// ```
/// use oxirag::layer1_echo::MetadataFilter;
/// use std::collections::HashMap;
///
/// // Simple equality filter
/// let filter = MetadataFilter::eq("category", "science");
///
/// // Complex filter with AND/OR
/// let complex = MetadataFilter::and(vec![
///     MetadataFilter::eq("status", "published"),
///     MetadataFilter::or(vec![
///         MetadataFilter::eq("category", "science"),
///         MetadataFilter::eq("category", "technology"),
///     ]),
/// ]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetadataFilter {
    /// Field equals value.
    Eq {
        /// The metadata field name.
        field: String,
        /// The value to compare against.
        value: String,
    },
    /// Field not equals value.
    Ne {
        /// The metadata field name.
        field: String,
        /// The value to compare against.
        value: String,
    },
    /// Field contains substring.
    Contains {
        /// The metadata field name.
        field: String,
        /// The substring to search for.
        substring: String,
    },
    /// Field exists (is present in metadata).
    Exists {
        /// The metadata field name.
        field: String,
    },
    /// Field does not exist (is not present in metadata).
    NotExists {
        /// The metadata field name.
        field: String,
    },
    /// All filters must match (logical AND).
    And(Vec<MetadataFilter>),
    /// Any filter must match (logical OR).
    Or(Vec<MetadataFilter>),
}

impl MetadataFilter {
    /// Check if the metadata matches this filter.
    ///
    /// # Arguments
    ///
    /// * `metadata` - The document metadata to check against.
    ///
    /// # Returns
    ///
    /// `true` if the metadata matches the filter, `false` otherwise.
    #[must_use]
    pub fn matches(&self, metadata: &HashMap<String, String>) -> bool {
        match self {
            Self::Eq { field, value } => metadata.get(field).is_some_and(|v| v == value),
            Self::Ne { field, value } => metadata.get(field).is_none_or(|v| v != value),
            Self::Contains { field, substring } => {
                metadata.get(field).is_some_and(|v| v.contains(substring))
            }
            Self::Exists { field } => metadata.contains_key(field),
            Self::NotExists { field } => !metadata.contains_key(field),
            Self::And(filters) => filters.iter().all(|f| f.matches(metadata)),
            Self::Or(filters) => {
                if filters.is_empty() {
                    true
                } else {
                    filters.iter().any(|f| f.matches(metadata))
                }
            }
        }
    }

    /// Create an equality filter.
    ///
    /// # Arguments
    ///
    /// * `field` - The metadata field name.
    /// * `value` - The value to compare against.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirag::layer1_echo::MetadataFilter;
    /// use std::collections::HashMap;
    ///
    /// let filter = MetadataFilter::eq("category", "science");
    /// let mut metadata = HashMap::new();
    /// metadata.insert("category".to_string(), "science".to_string());
    /// assert!(filter.matches(&metadata));
    /// ```
    #[must_use]
    pub fn eq(field: impl Into<String>, value: impl Into<String>) -> Self {
        Self::Eq {
            field: field.into(),
            value: value.into(),
        }
    }

    /// Create a not-equals filter.
    ///
    /// # Arguments
    ///
    /// * `field` - The metadata field name.
    /// * `value` - The value to compare against.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirag::layer1_echo::MetadataFilter;
    /// use std::collections::HashMap;
    ///
    /// let filter = MetadataFilter::ne("status", "draft");
    /// let mut metadata = HashMap::new();
    /// metadata.insert("status".to_string(), "published".to_string());
    /// assert!(filter.matches(&metadata));
    /// ```
    #[must_use]
    pub fn ne(field: impl Into<String>, value: impl Into<String>) -> Self {
        Self::Ne {
            field: field.into(),
            value: value.into(),
        }
    }

    /// Create a contains filter.
    ///
    /// # Arguments
    ///
    /// * `field` - The metadata field name.
    /// * `substring` - The substring to search for.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirag::layer1_echo::MetadataFilter;
    /// use std::collections::HashMap;
    ///
    /// let filter = MetadataFilter::contains("title", "Rust");
    /// let mut metadata = HashMap::new();
    /// metadata.insert("title".to_string(), "Learning Rust Programming".to_string());
    /// assert!(filter.matches(&metadata));
    /// ```
    #[must_use]
    pub fn contains(field: impl Into<String>, substring: impl Into<String>) -> Self {
        Self::Contains {
            field: field.into(),
            substring: substring.into(),
        }
    }

    /// Create an exists filter.
    ///
    /// # Arguments
    ///
    /// * `field` - The metadata field name.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirag::layer1_echo::MetadataFilter;
    /// use std::collections::HashMap;
    ///
    /// let filter = MetadataFilter::exists("author");
    /// let mut metadata = HashMap::new();
    /// metadata.insert("author".to_string(), "John Doe".to_string());
    /// assert!(filter.matches(&metadata));
    /// ```
    #[must_use]
    pub fn exists(field: impl Into<String>) -> Self {
        Self::Exists {
            field: field.into(),
        }
    }

    /// Create a not-exists filter.
    ///
    /// # Arguments
    ///
    /// * `field` - The metadata field name.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirag::layer1_echo::MetadataFilter;
    /// use std::collections::HashMap;
    ///
    /// let filter = MetadataFilter::not_exists("deprecated");
    /// let metadata = HashMap::new();
    /// assert!(filter.matches(&metadata));
    /// ```
    #[must_use]
    pub fn not_exists(field: impl Into<String>) -> Self {
        Self::NotExists {
            field: field.into(),
        }
    }

    /// Create a logical AND filter.
    ///
    /// All provided filters must match for the AND filter to match.
    ///
    /// # Arguments
    ///
    /// * `filters` - The filters to combine with AND.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirag::layer1_echo::MetadataFilter;
    /// use std::collections::HashMap;
    ///
    /// let filter = MetadataFilter::and(vec![
    ///     MetadataFilter::eq("status", "published"),
    ///     MetadataFilter::eq("category", "science"),
    /// ]);
    ///
    /// let mut metadata = HashMap::new();
    /// metadata.insert("status".to_string(), "published".to_string());
    /// metadata.insert("category".to_string(), "science".to_string());
    /// assert!(filter.matches(&metadata));
    /// ```
    #[must_use]
    pub fn and(filters: Vec<MetadataFilter>) -> Self {
        Self::And(filters)
    }

    /// Create a logical OR filter.
    ///
    /// At least one of the provided filters must match for the OR filter to match.
    /// An empty OR filter matches everything.
    ///
    /// # Arguments
    ///
    /// * `filters` - The filters to combine with OR.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirag::layer1_echo::MetadataFilter;
    /// use std::collections::HashMap;
    ///
    /// let filter = MetadataFilter::or(vec![
    ///     MetadataFilter::eq("category", "science"),
    ///     MetadataFilter::eq("category", "technology"),
    /// ]);
    ///
    /// let mut metadata = HashMap::new();
    /// metadata.insert("category".to_string(), "science".to_string());
    /// assert!(filter.matches(&metadata));
    /// ```
    #[must_use]
    pub fn or(filters: Vec<MetadataFilter>) -> Self {
        Self::Or(filters)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_metadata(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect()
    }

    #[test]
    fn test_eq_filter() {
        let filter = MetadataFilter::eq("category", "science");

        let matching = make_metadata(&[("category", "science")]);
        let non_matching = make_metadata(&[("category", "technology")]);
        let missing = make_metadata(&[("other", "value")]);

        assert!(filter.matches(&matching));
        assert!(!filter.matches(&non_matching));
        assert!(!filter.matches(&missing));
    }

    #[test]
    fn test_ne_filter() {
        let filter = MetadataFilter::ne("status", "draft");

        let matching_different = make_metadata(&[("status", "published")]);
        let matching_missing = make_metadata(&[("other", "value")]);
        let non_matching = make_metadata(&[("status", "draft")]);

        assert!(filter.matches(&matching_different));
        assert!(filter.matches(&matching_missing));
        assert!(!filter.matches(&non_matching));
    }

    #[test]
    fn test_contains_filter() {
        let filter = MetadataFilter::contains("title", "Rust");

        let matching = make_metadata(&[("title", "Learning Rust Programming")]);
        let non_matching = make_metadata(&[("title", "Python Guide")]);
        let missing = make_metadata(&[("other", "value")]);

        assert!(filter.matches(&matching));
        assert!(!filter.matches(&non_matching));
        assert!(!filter.matches(&missing));
    }

    #[test]
    fn test_exists_filter() {
        let filter = MetadataFilter::exists("author");

        let matching = make_metadata(&[("author", "John Doe")]);
        let non_matching = make_metadata(&[("title", "Book")]);

        assert!(filter.matches(&matching));
        assert!(!filter.matches(&non_matching));
    }

    #[test]
    fn test_not_exists_filter() {
        let filter = MetadataFilter::not_exists("deprecated");

        let matching = make_metadata(&[("status", "active")]);
        let non_matching = make_metadata(&[("deprecated", "true")]);

        assert!(filter.matches(&matching));
        assert!(!filter.matches(&non_matching));
    }

    #[test]
    fn test_and_filter() {
        let filter = MetadataFilter::and(vec![
            MetadataFilter::eq("status", "published"),
            MetadataFilter::eq("category", "science"),
        ]);

        let matching = make_metadata(&[("status", "published"), ("category", "science")]);
        let partial1 = make_metadata(&[("status", "published"), ("category", "tech")]);
        let partial2 = make_metadata(&[("status", "draft"), ("category", "science")]);

        assert!(filter.matches(&matching));
        assert!(!filter.matches(&partial1));
        assert!(!filter.matches(&partial2));
    }

    #[test]
    fn test_and_filter_empty() {
        let filter = MetadataFilter::and(vec![]);
        let metadata = make_metadata(&[("any", "value")]);
        assert!(filter.matches(&metadata));
    }

    #[test]
    fn test_or_filter() {
        let filter = MetadataFilter::or(vec![
            MetadataFilter::eq("category", "science"),
            MetadataFilter::eq("category", "technology"),
        ]);

        let matching1 = make_metadata(&[("category", "science")]);
        let matching2 = make_metadata(&[("category", "technology")]);
        let non_matching = make_metadata(&[("category", "art")]);

        assert!(filter.matches(&matching1));
        assert!(filter.matches(&matching2));
        assert!(!filter.matches(&non_matching));
    }

    #[test]
    fn test_or_filter_empty() {
        let filter = MetadataFilter::or(vec![]);
        let metadata = make_metadata(&[("any", "value")]);
        assert!(filter.matches(&metadata));
    }

    #[test]
    fn test_complex_nested_filter() {
        // (status == "published") AND (category == "science" OR category == "tech")
        let filter = MetadataFilter::and(vec![
            MetadataFilter::eq("status", "published"),
            MetadataFilter::or(vec![
                MetadataFilter::eq("category", "science"),
                MetadataFilter::eq("category", "tech"),
            ]),
        ]);

        let matching1 = make_metadata(&[("status", "published"), ("category", "science")]);
        let matching2 = make_metadata(&[("status", "published"), ("category", "tech")]);
        let non_matching1 = make_metadata(&[("status", "draft"), ("category", "science")]);
        let non_matching2 = make_metadata(&[("status", "published"), ("category", "art")]);

        assert!(filter.matches(&matching1));
        assert!(filter.matches(&matching2));
        assert!(!filter.matches(&non_matching1));
        assert!(!filter.matches(&non_matching2));
    }

    #[test]
    fn test_filter_equality() {
        let filter1 = MetadataFilter::eq("field", "value");
        let filter2 = MetadataFilter::eq("field", "value");
        let filter3 = MetadataFilter::eq("field", "other");

        assert_eq!(filter1, filter2);
        assert_ne!(filter1, filter3);
    }

    #[test]
    fn test_filter_clone() {
        let original = MetadataFilter::and(vec![
            MetadataFilter::eq("a", "b"),
            MetadataFilter::or(vec![MetadataFilter::exists("c")]),
        ]);

        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn test_filter_debug() {
        let filter = MetadataFilter::eq("field", "value");
        let debug_str = format!("{filter:?}");
        assert!(debug_str.contains("Eq"));
        assert!(debug_str.contains("field"));
        assert!(debug_str.contains("value"));
    }
}
