//! Integration tests for snapshot serialization functionality.

#[cfg(feature = "serde")]
mod serde_tests {
    use grafial_core::engine::errors::ExecError;
    use grafial_core::engine::graph::{BeliefGraph, EdgeId, NodeId};
    use grafial_core::engine::snapshot::{
        load_snapshot_binary, load_snapshot_json, save_snapshot_binary, save_snapshot_json,
        Snapshot,
    };

    fn create_test_graph() -> BeliefGraph {
        let mut g = BeliefGraph::default();
        let n1 = g.add_node("Person".to_string());
        let n2 = g.add_node("Person".to_string());
        let e1 = g.add_edge(n1, "REL".to_string(), n2);

        // Apply some observations
        g.observe_edge(e1, true).unwrap();
        g.observe_attr(n1, "value".to_string(), 1.0).unwrap();

        // Ensure all deltas are applied (snapshot will do this too, but be explicit)
        g
    }

    #[test]
    fn test_snapshot_creation() {
        let graph = create_test_graph();
        let snapshot = Snapshot::new(graph, None);

        assert_eq!(snapshot.metadata.version, env!("CARGO_PKG_VERSION"));
        assert!(snapshot.metadata.features.contains(&"serde".to_string()));
        assert_eq!(snapshot.metadata.registry_hash, None);
        assert_eq!(snapshot.graph.nodes().len(), 2);
        assert_eq!(snapshot.graph.edges().len(), 1);
    }

    #[test]
    fn test_snapshot_json_roundtrip() {
        let graph = create_test_graph();
        let snapshot = Snapshot::new(graph, Some("test_hash".to_string()));

        // Save to JSON
        let json = save_snapshot_json(&snapshot).unwrap();
        assert!(json.contains("graph"));
        assert!(json.contains("metadata"));
        assert!(json.contains("test_hash"));

        // Load from JSON
        let loaded = load_snapshot_json(&json).unwrap();

        // Verify graph structure
        assert_eq!(snapshot.graph.nodes().len(), loaded.graph.nodes().len());
        assert_eq!(snapshot.graph.edges().len(), loaded.graph.edges().len());

        // Verify metadata
        assert_eq!(snapshot.metadata.version, loaded.metadata.version);
        assert_eq!(
            snapshot.metadata.registry_hash,
            loaded.metadata.registry_hash
        );
    }

    #[test]
    fn test_snapshot_binary_roundtrip() {
        let graph = create_test_graph();
        let snapshot = Snapshot::new(graph, Some("test_hash".to_string()));

        // Save to binary
        let binary = save_snapshot_binary(&snapshot).unwrap();
        assert!(!binary.is_empty());

        // Load from binary
        let loaded = load_snapshot_binary(&binary).unwrap();

        // Verify graph structure
        assert_eq!(snapshot.graph.nodes().len(), loaded.graph.nodes().len());
        assert_eq!(snapshot.graph.edges().len(), loaded.graph.edges().len());

        // Verify metadata
        assert_eq!(snapshot.metadata.version, loaded.metadata.version);
        assert_eq!(
            snapshot.metadata.registry_hash,
            loaded.metadata.registry_hash
        );
    }

    #[test]
    fn test_snapshot_compatibility_validation() {
        let graph = create_test_graph();
        let mut snapshot = Snapshot::new(graph, None);

        // Should pass validation with current version
        assert!(snapshot.validate_compatibility().is_ok());

        // Change version to invalid value
        snapshot.metadata.version = "0.0.0-invalid".to_string();

        // Should fail validation
        let result = snapshot.validate_compatibility();
        assert!(result.is_err());
        if let Err(ExecError::ValidationError(msg)) = result {
            assert!(msg.contains("version mismatch"));
        } else {
            panic!("Expected ValidationError");
        }
    }

    #[test]
    fn test_snapshot_applies_deltas() {
        let mut graph = create_test_graph();

        // Add a node that will be in delta
        let n3 = graph.add_node("Person".to_string());

        // Create snapshot - should apply deltas
        let snapshot = Snapshot::new(graph, None);

        // The new node should be in the snapshot
        assert_eq!(snapshot.graph.nodes().len(), 3);
        assert!(snapshot.graph.node(n3).is_some());
    }
}
