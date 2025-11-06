//! Snapshot and serialization functionality.
//!
//! Provides checkpoint/save functionality for BeliefGraph instances with
//! version metadata and compatibility validation.

use crate::engine::errors::ExecError;
use crate::engine::graph::BeliefGraph;

/// Metadata included in snapshots for compatibility checking.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SnapshotMetadata {
    /// Engine version string
    pub version: String,
    /// Feature flags enabled when snapshot was created
    pub features: Vec<String>,
    /// Registry hash for metric compatibility checking
    pub registry_hash: Option<String>,
}

/// A snapshot of a BeliefGraph with metadata.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Snapshot {
    /// The graph state
    pub graph: BeliefGraph,
    /// Metadata for compatibility checking
    pub metadata: SnapshotMetadata,
}

impl Snapshot {
    /// Creates a new snapshot from a graph.
    ///
    /// # Arguments
    ///
    /// * `graph` - The graph to snapshot (applies any pending deltas first)
    /// * `registry_hash` - Optional hash of metric registry for compatibility checking
    ///
    /// # Returns
    ///
    /// A new Snapshot with current engine version and feature flags
    pub fn new(mut graph: BeliefGraph, registry_hash: Option<String>) -> Self {
        // Ensure graph has no pending deltas before snapshotting
        graph.ensure_owned();
        
        let metadata = SnapshotMetadata {
            version: env!("CARGO_PKG_VERSION").to_string(),
            features: get_enabled_features(),
            registry_hash,
        };
        
        Self { graph, metadata }
    }

    /// Validates that this snapshot is compatible with the current engine.
    ///
    /// Checks version compatibility and feature flags. Returns Ok(()) if compatible,
    /// Err(ExecError) if incompatible.
    pub fn validate_compatibility(&self) -> Result<(), ExecError> {
        let current_version = env!("CARGO_PKG_VERSION");
        
        // For now, require exact version match
        // TODO: Implement semantic versioning compatibility checking
        if self.metadata.version != current_version {
            return Err(ExecError::ValidationError(
                format!(
                    "Snapshot version mismatch: snapshot was created with version {}, current version is {}",
                    self.metadata.version, current_version
                )
            ));
        }
        
        // Check that all required features are available
        let current_features = get_enabled_features();
        for required_feature in &self.metadata.features {
            if !current_features.contains(required_feature) {
                return Err(ExecError::ValidationError(
                    format!(
                        "Snapshot requires feature '{}' which is not enabled",
                        required_feature
                    )
                ));
            }
        }
        
        Ok(())
    }
}

/// Returns a list of enabled feature flags.
fn get_enabled_features() -> Vec<String> {
    #[allow(unused_mut)]  // mut is needed when features are enabled
    let mut features = Vec::new();
    
    #[cfg(feature = "rayon")]
    {
        features.push("rayon".to_string());
    }
    
    #[cfg(feature = "serde")]
    {
        features.push("serde".to_string());
    }
    
    #[cfg(feature = "tracing")]
    {
        features.push("tracing".to_string());
    }
    
    features
}

/// Saves a snapshot to a JSON string.
///
/// # Arguments
///
/// * `snapshot` - The snapshot to save
///
/// # Returns
///
/// * `Ok(String)` - JSON string representation
/// * `Err(ExecError)` - If serialization fails
#[cfg(feature = "serde")]
pub fn save_snapshot_json(snapshot: &Snapshot) -> Result<String, ExecError> {
    serde_json::to_string_pretty(snapshot)
        .map_err(|e| ExecError::Internal(format!("Failed to serialize snapshot: {}", e)))
}

/// Loads a snapshot from a JSON string.
///
/// # Arguments
///
/// * `json` - JSON string representation
///
/// # Returns
///
/// * `Ok(Snapshot)` - The loaded snapshot
/// * `Err(ExecError)` - If deserialization fails or snapshot is incompatible
#[cfg(feature = "serde")]
pub fn load_snapshot_json(json: &str) -> Result<Snapshot, ExecError> {
    let snapshot: Snapshot = serde_json::from_str(json)
        .map_err(|e| ExecError::Internal(format!("Failed to deserialize snapshot: {}", e)))?;
    
    snapshot.validate_compatibility()?;
    Ok(snapshot)
}

/// Saves a snapshot to a binary format (bincode).
///
/// # Arguments
///
/// * `snapshot` - The snapshot to save
///
/// # Returns
///
/// * `Ok(Vec<u8>)` - Binary representation
/// * `Err(ExecError)` - If serialization fails
#[cfg(feature = "serde")]
pub fn save_snapshot_binary(snapshot: &Snapshot) -> Result<Vec<u8>, ExecError> {
    bincode::serialize(snapshot)
        .map_err(|e| ExecError::Internal(format!("Failed to serialize snapshot: {}", e)))
}

/// Loads a snapshot from binary format (bincode).
///
/// # Arguments
///
/// * `data` - Binary representation
///
/// # Returns
///
/// * `Ok(Snapshot)` - The loaded snapshot
/// * `Err(ExecError)` - If deserialization fails or snapshot is incompatible
#[cfg(feature = "serde")]
pub fn load_snapshot_binary(data: &[u8]) -> Result<Snapshot, ExecError> {
    let snapshot: Snapshot = bincode::deserialize(data)
        .map_err(|e| ExecError::Internal(format!("Failed to deserialize snapshot: {}", e)))?;
    
    snapshot.validate_compatibility()?;
    Ok(snapshot)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::graph::BeliefGraph;

    fn create_test_graph() -> BeliefGraph {
        use std::collections::HashMap;
        use crate::engine::graph::BetaPosterior;
        let mut g = BeliefGraph::default();
        let n1 = g.add_node("Person".to_string(), HashMap::new());
        let n2 = g.add_node("Person".to_string(), HashMap::new());
        let beta = BetaPosterior { alpha: 1.0, beta: 1.0 };
        let _e1 = g.add_edge(n1, n2, "REL".to_string(), beta);
        g
    }

    #[test]
    fn test_snapshot_new_without_registry_hash() {
        let graph = create_test_graph();
        let snapshot = Snapshot::new(graph, None);
        
        assert_eq!(snapshot.metadata.version, env!("CARGO_PKG_VERSION"));
        assert_eq!(snapshot.metadata.registry_hash, None);
        assert_eq!(snapshot.graph.nodes().len(), 2);
        assert_eq!(snapshot.graph.edges().len(), 1);
    }

    #[test]
    fn test_snapshot_new_with_registry_hash() {
        let graph = create_test_graph();
        let hash = Some("abc123".to_string());
        let snapshot = Snapshot::new(graph, hash.clone());
        
        assert_eq!(snapshot.metadata.version, env!("CARGO_PKG_VERSION"));
        assert_eq!(snapshot.metadata.registry_hash, hash);
    }

    #[test]
    fn test_snapshot_new_applies_pending_deltas() {
        use std::collections::HashMap;
        use crate::engine::graph::GaussianPosterior;
        let mut graph = BeliefGraph::default();
        let mut attrs = HashMap::new();
        attrs.insert("score".to_string(), GaussianPosterior { mean: 0.0, precision: 0.01 });
        let n1 = graph.add_node("Person".to_string(), attrs);
        let node_id = n1;
        
        // Make a mutation that creates a delta
        graph.set_expectation(node_id, "score", 10.0).unwrap();
        
        // Create snapshot - should apply delta
        let snapshot = Snapshot::new(graph, None);
        
        // Graph should be accessible and have the update
        let value = snapshot.graph.expectation(node_id, "score").unwrap();
        assert_eq!(value, 10.0);
    }

    #[test]
    fn test_snapshot_metadata_contains_enabled_features() {
        let graph = create_test_graph();
        let _snapshot = Snapshot::new(graph, None);
        
        // Should always contain at least the features we enable
        // (serde is enabled in tests, so we should see it)
        #[cfg(feature = "serde")]
        assert!(_snapshot.metadata.features.contains(&"serde".to_string()));
    }

    #[test]
    fn test_validate_compatibility_succeeds_for_current_version() {
        let graph = create_test_graph();
        let snapshot = Snapshot::new(graph, None);
        
        // Should succeed for snapshot created with current version
        assert!(snapshot.validate_compatibility().is_ok());
    }

    #[test]
    fn test_validate_compatibility_fails_for_version_mismatch() {
        let graph = create_test_graph();
        let mut snapshot = Snapshot::new(graph, None);
        
        // Change version to something different
        snapshot.metadata.version = "0.99.0".to_string();
        
        let result = snapshot.validate_compatibility();
        assert!(result.is_err());
        
        if let Err(ExecError::ValidationError(msg)) = result {
            assert!(msg.contains("version mismatch"));
            assert!(msg.contains("0.99.0"));
            assert!(msg.contains(env!("CARGO_PKG_VERSION")));
        } else {
            panic!("Expected ValidationError");
        }
    }

    #[test]
    fn test_validate_compatibility_fails_for_missing_feature() {
        let graph = create_test_graph();
        let mut snapshot = Snapshot::new(graph, None);
        
        // Add a feature that doesn't exist
        snapshot.metadata.features.push("nonexistent_feature".to_string());
        
        let result = snapshot.validate_compatibility();
        assert!(result.is_err());
        
        if let Err(ExecError::ValidationError(msg)) = result {
            assert!(msg.contains("requires feature"));
            assert!(msg.contains("nonexistent_feature"));
            assert!(msg.contains("not enabled"));
        } else {
            panic!("Expected ValidationError");
        }
    }

    #[test]
    fn test_validate_compatibility_succeeds_with_subset_of_features() {
        let graph = create_test_graph();
        let mut snapshot = Snapshot::new(graph, None);
        
        // Remove all features (snapshot created with no features)
        snapshot.metadata.features.clear();
        
        // Should still validate (snapshot doesn't require any features)
        assert!(snapshot.validate_compatibility().is_ok());
    }

    #[test]
    fn test_validate_compatibility_succeeds_with_available_features() {
        let graph = create_test_graph();
        let snapshot = Snapshot::new(graph, None);
        
        // Create a new snapshot with same features as current
        // Just use the features from the original snapshot
        let compatible_snapshot = Snapshot {
            graph: snapshot.graph.clone(),
            metadata: SnapshotMetadata {
                version: env!("CARGO_PKG_VERSION").to_string(),
                features: snapshot.metadata.features.clone(),
                registry_hash: None,
            },
        };
        
        assert!(compatible_snapshot.validate_compatibility().is_ok());
    }

    #[test]
    fn test_get_enabled_features() {
        // Test that snapshot includes features
        let graph = create_test_graph();
        let snapshot = Snapshot::new(graph, None);
        let features = &snapshot.metadata.features;
        
        // Features should be a Vec<String>
        assert!(features.iter().all(|f| f.chars().all(|c| c.is_alphanumeric() || c == '_')));
        
        // Should include serde if feature is enabled
        #[cfg(feature = "serde")]
        assert!(features.contains(&"serde".to_string()));
    }
}
