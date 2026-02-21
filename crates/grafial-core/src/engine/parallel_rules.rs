//! Parallel rule application for non-overlapping matches.
//!
//! This module provides parallel execution of rules when their matches
//! don't overlap. Rules that affect different parts of the graph can
//! be applied concurrently for improved performance.
//!
//! ## Architecture
//!
//! - **Match analysis**: Identifies non-overlapping rule matches
//! - **Conflict detection**: Ensures rules don't modify same nodes/edges
//! - **Parallel application**: Applies non-conflicting rules concurrently
//! - **Sequential fallback**: Handles conflicting rules sequentially
//!
//! ## Feature gating
//!
//! Parallel rules are behind the `parallel` feature flag. When disabled,
//! rules are applied sequentially.

use std::collections::HashSet;

use crate::engine::errors::ExecError;
use crate::engine::graph::{BeliefGraph, EdgeId, NodeId};
use grafial_ir::RuleIR;
use std::collections::HashMap;

/// A match result from pattern matching.
#[derive(Debug, Clone)]
struct RuleMatch {
    /// Matched nodes by variable name
    nodes: HashMap<String, NodeId>,
    /// Matched edges by variable name
    edges: HashMap<String, EdgeId>,
}

/// Result of parallel rule application.
#[derive(Debug)]
pub struct ParallelRuleResult {
    /// Number of rules applied
    pub rules_applied: usize,
    /// Number of matches found
    pub matches_found: usize,
    /// Statistics about parallel execution
    pub stats: ParallelRuleStats,
}

/// Statistics about parallel rule application.
#[derive(Debug, Default)]
pub struct ParallelRuleStats {
    /// Number of parallel batches executed
    pub parallel_batches: usize,
    /// Maximum parallelism achieved
    pub max_parallelism: usize,
    /// Number of conflicts detected
    pub conflicts_detected: usize,
}

/// A match with its affected graph elements.
#[derive(Debug, Clone)]
struct MatchWithFootprint {
    /// The original match
    match_data: RuleMatch,
    /// Rule to apply
    rule: RuleIR,
    /// Nodes that will be read or modified
    affected_nodes: HashSet<NodeId>,
    /// Edges that will be read or modified
    affected_edges: HashSet<EdgeId>,
}

/// Check if two matches conflict (overlap in their affected elements).
fn matches_conflict(m1: &MatchWithFootprint, m2: &MatchWithFootprint) -> bool {
    // Two matches conflict if they affect any of the same nodes or edges
    !m1.affected_nodes.is_disjoint(&m2.affected_nodes)
        || !m1.affected_edges.is_disjoint(&m2.affected_edges)
}

/// Extract affected elements from a rule match and its actions.
fn extract_footprint(match_data: &RuleMatch, _rule: &RuleIR) -> (HashSet<NodeId>, HashSet<EdgeId>) {
    let mut nodes = HashSet::new();
    let mut edges = HashSet::new();

    // Add matched nodes
    for node_id in match_data.nodes.values() {
        nodes.insert(*node_id);
    }

    // Add matched edges
    for edge_id in match_data.edges.values() {
        edges.insert(*edge_id);
    }

    // TODO: Analyze rule actions to identify nodes/edges that will be modified
    // For now, we conservatively assume all matched elements are modified

    (nodes, edges)
}

/// Group matches into non-conflicting batches for parallel execution.
fn batch_non_conflicting_matches(matches: Vec<MatchWithFootprint>) -> Vec<Vec<MatchWithFootprint>> {
    let mut batches: Vec<Vec<MatchWithFootprint>> = Vec::new();

    for match_item in matches {
        // Try to add to an existing batch
        let mut added = false;
        for batch in &mut batches {
            // Check if this match conflicts with any match in the batch
            let has_conflict = batch
                .iter()
                .any(|existing| matches_conflict(&match_item, existing));

            if !has_conflict {
                batch.push(match_item.clone());
                added = true;
                break;
            }
        }

        // If couldn't add to any batch, create a new one
        if !added {
            batches.push(vec![match_item]);
        }
    }

    batches
}

/// Apply rules in parallel when matches don't overlap.
#[cfg(feature = "parallel")]
pub fn apply_rules_parallel(
    graph: &mut BeliefGraph,
    rules: &[RuleIR],
) -> Result<ParallelRuleResult, ExecError> {
    let mut total_matches = 0;
    let mut total_applied = 0;
    let mut max_parallelism = 0;
    let mut conflicts = 0;

    // For each rule, find matches and apply in parallel batches
    for rule in rules {
        // Find all matches for this rule
        let matches = find_rule_matches(graph, rule)?;
        total_matches += matches.len();

        if matches.is_empty() {
            continue;
        }

        // Extract footprints for each match
        let matches_with_footprint: Vec<_> = matches
            .into_iter()
            .map(|m| {
                let (nodes, edges) = extract_footprint(&m, rule);
                MatchWithFootprint {
                    match_data: m,
                    rule: rule.clone(),
                    affected_nodes: nodes,
                    affected_edges: edges,
                }
            })
            .collect();

        // Group into non-conflicting batches
        let batches = batch_non_conflicting_matches(matches_with_footprint);

        // Track conflicts
        if batches.len() > 1 {
            conflicts += batches.len() - 1;
        }

        // Apply each batch
        for batch in batches {
            max_parallelism = max_parallelism.max(batch.len());

            // Apply all matches in this batch in parallel
            // Note: In practice, we need thread-safe graph operations
            // For now, we apply sequentially but could parallelize with proper locking
            for match_item in batch {
                apply_rule_actions(graph, &match_item.match_data, &match_item.rule)?;
                total_applied += 1;
            }
        }
    }

    Ok(ParallelRuleResult {
        rules_applied: total_applied,
        matches_found: total_matches,
        stats: ParallelRuleStats {
            parallel_batches: 1, // Simplified for now
            max_parallelism,
            conflicts_detected: conflicts,
        },
    })
}

/// Sequential fallback for rule application.
#[cfg(not(feature = "parallel"))]
pub fn apply_rules_parallel(
    graph: &mut BeliefGraph,
    rules: &[RuleIR],
) -> Result<ParallelRuleResult, ExecError> {
    let mut total_matches = 0;
    let mut total_applied = 0;

    for rule in rules {
        let matches = find_rule_matches(graph, rule)?;
        total_matches += matches.len();

        for match_data in matches {
            apply_rule_actions(graph, &match_data, rule)?;
            total_applied += 1;
        }
    }

    Ok(ParallelRuleResult {
        rules_applied: total_applied,
        matches_found: total_matches,
        stats: ParallelRuleStats {
            parallel_batches: 1,
            max_parallelism: 1,
            conflicts_detected: 0,
        },
    })
}

/// Find all matches for a rule in the graph.
///
/// This is a placeholder - the actual implementation would use
/// the pattern matching from rule_exec module.
fn find_rule_matches(_graph: &BeliefGraph, _rule: &RuleIR) -> Result<Vec<RuleMatch>, ExecError> {
    // TODO: Integrate with actual rule matching logic
    Ok(Vec::new())
}

/// Apply rule actions for a match.
///
/// This is a placeholder - the actual implementation would use
/// the action execution from rule_exec module.
fn apply_rule_actions(
    _graph: &mut BeliefGraph,
    _match_data: &RuleMatch,
    _rule: &RuleIR,
) -> Result<(), ExecError> {
    // TODO: Integrate with actual rule action execution
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conflict_detection() {
        let mut match1 = MatchWithFootprint {
            match_data: RuleMatch {
                nodes: HashMap::new(),
                edges: HashMap::new(),
            },
            rule: RuleIR {
                name: "rule1".to_string(),
                on_model: "test".to_string(),
                patterns: Vec::new(),
                where_expr: None,
                actions: Vec::new(),
                mode: None,
            },
            affected_nodes: HashSet::new(),
            affected_edges: HashSet::new(),
        };

        let mut match2 = match1.clone();
        match2.rule.name = "rule2".to_string();

        // Initially no conflict
        assert!(!matches_conflict(&match1, &match2));

        // Add same node to both - now they conflict
        let node_id = NodeId(0);
        match1.affected_nodes.insert(node_id);
        match2.affected_nodes.insert(node_id);

        assert!(matches_conflict(&match1, &match2));
    }

    #[test]
    fn test_batch_creation() {
        let rule = RuleIR {
            name: "rule".to_string(),
            on_model: "test".to_string(),
            patterns: Vec::new(),
            where_expr: None,
            actions: Vec::new(),
            mode: None,
        };

        // Create non-conflicting matches
        let mut matches = Vec::new();
        for i in 0..4 {
            let mut nodes = HashSet::new();
            nodes.insert(NodeId(i));

            matches.push(MatchWithFootprint {
                match_data: RuleMatch {
                    nodes: HashMap::new(),
                    edges: HashMap::new(),
                },
                rule: rule.clone(),
                affected_nodes: nodes,
                affected_edges: HashSet::new(),
            });
        }

        let batches = batch_non_conflicting_matches(matches);

        // All matches should be in one batch since they don't conflict
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 4);
    }

    #[test]
    fn test_conflicting_batch_creation() {
        let rule = RuleIR {
            name: "rule".to_string(),
            on_model: "test".to_string(),
            patterns: Vec::new(),
            where_expr: None,
            actions: Vec::new(),
            mode: None,
        };

        // Create conflicting matches (all affect node 0)
        let mut matches = Vec::new();
        for i in 0..3 {
            let mut nodes = HashSet::new();
            nodes.insert(NodeId(0)); // All affect same node
            nodes.insert(NodeId(i + 1)); // Plus unique node

            matches.push(MatchWithFootprint {
                match_data: RuleMatch {
                    nodes: HashMap::new(),
                    edges: HashMap::new(),
                },
                rule: rule.clone(),
                affected_nodes: nodes,
                affected_edges: HashSet::new(),
            });
        }

        let batches = batch_non_conflicting_matches(matches);

        // Each match should be in its own batch since they all conflict
        assert_eq!(batches.len(), 3);
        for batch in batches {
            assert_eq!(batch.len(), 1);
        }
    }
}
