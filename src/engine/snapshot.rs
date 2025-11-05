//! Snapshot and serialization functionality.
//!
//! Provides checkpoint/save functionality for BeliefGraph instances with
//! version metadata and compatibility validation.

use std::collections::HashMap;

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
    let mut features = Vec::new();
    
    #[cfg(feature = "rayon")]
    features.push("rayon".to_string());
    
    #[cfg(feature = "serde")]
    features.push("serde".to_string());
    
    #[cfg(feature = "tracing")]
    features.push("tracing".to_string());
    
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

