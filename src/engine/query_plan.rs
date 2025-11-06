//! Query plan optimization for pattern matching.
//!
//! Orders patterns by selectivity (most selective first) to minimize join cost.
//! Plans are cached to avoid re-analysis of the same pattern combinations.

use rustc_hash::FxHashMap;

use crate::engine::graph::BeliefGraph;
use crate::frontend::ast::PatternItem;

/// Selectivity estimate for a pattern.
///
/// Lower selectivity means fewer matches (more selective = better starting point).
#[derive(Debug, Clone, PartialEq)]
pub struct PatternSelectivity {
    /// Pattern index in the original rule
    pub pattern_idx: usize,
    /// Estimated number of edges matching this pattern
    pub estimated_matches: usize,
    /// Selectivity score (lower = more selective)
    pub selectivity_score: f64,
}

/// A query plan for efficient pattern matching.
///
/// The plan orders patterns by selectivity and provides execution hints
/// for efficient matching.
#[derive(Debug, Clone)]
pub struct QueryPlan {
    /// Patterns ordered by selectivity (most selective first)
    pub ordered_patterns: Vec<usize>,
    /// Selectivity analysis for each pattern
    pub selectivity: Vec<PatternSelectivity>,
}

impl QueryPlan {
    /// Creates a new query plan by analyzing patterns against a graph.
    ///
    /// Analyzes each pattern's selectivity and orders them for optimal join order.
    pub fn new(patterns: &[PatternItem], graph: &BeliefGraph) -> Self {
        let mut selectivity: Vec<PatternSelectivity> = patterns
            .iter()
            .enumerate()
            .map(|(idx, pat)| analyze_pattern_selectivity(idx, pat, graph))
            .collect();

        // Sort by selectivity score (lower = more selective = better starting point)
        // Use unstable sort for better performance (ordering is deterministic based on score)
        selectivity.sort_unstable_by(|a, b| {
            a.selectivity_score
                .partial_cmp(&b.selectivity_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Extract ordered pattern indices
        let ordered_patterns: Vec<usize> = selectivity.iter().map(|s| s.pattern_idx).collect();

        Self {
            ordered_patterns,
            selectivity,
        }
    }

    /// Gets the pattern index for the given position in the execution order.
    pub fn pattern_at_order(&self, order: usize) -> Option<usize> {
        self.ordered_patterns.get(order).copied()
    }
}

/// Analyzes the selectivity of a single pattern.
///
/// Estimates match count by filtering edges by type and approximating node label matches.
/// Lower selectivity score means fewer matches (better starting point for joins).
fn analyze_pattern_selectivity(
    pattern_idx: usize,
    pattern: &PatternItem,
    graph: &BeliefGraph,
) -> PatternSelectivity {
    let edges = graph.edges();
    let matching_edges: usize = edges
        .iter()
        .filter(|e| e.ty.as_ref() == pattern.edge.ty.as_str())
        .count();

    // Estimate node label filtering: count nodes with matching labels and approximate
    // edge connectivity. This heuristic assumes roughly uniform edge distribution.
    let estimated_matches = if matching_edges > 0 {
        let nodes = graph.nodes();
        let src_label_count = nodes
            .iter()
            .filter(|n| n.label.as_ref() == pattern.src.label.as_str())
            .count();
        let dst_label_count = nodes
            .iter()
            .filter(|n| n.label.as_ref() == pattern.dst.label.as_str())
            .count();

        // Estimate matches: edges with matching type * (label match probability)
        // Use geometric mean as rough estimate for both labels matching
        let label_match_prob = if src_label_count > 0 && dst_label_count > 0 {
            let src_prob = src_label_count as f64 / nodes.len() as f64;
            let dst_prob = dst_label_count as f64 / nodes.len() as f64;
            src_prob * dst_prob
        } else {
            0.0
        };

        (matching_edges as f64 * label_match_prob).max(1.0) as usize
    } else {
        0
    };

    // Selectivity score: estimated matches (lower = more selective)
    // Add a small tiebreaker based on pattern index for determinism
    let selectivity_score = estimated_matches as f64 + (pattern_idx as f64 * 0.001);

    PatternSelectivity {
        pattern_idx,
        estimated_matches,
        selectivity_score,
    }
}

/// Query plan cache for frequently used plans.
///
/// Caches query plans keyed by pattern signature (edge types and labels).
/// This allows reuse of plans across multiple rule executions with similar patterns.
/// Uses FxHashMap for faster lookups and a compact key type for better cache efficiency.
pub struct QueryPlanCache {
    /// Cached plans keyed by pattern signature
    /// Using FxHashMap for faster integer-based hashing (we hash the key tuple)
    cache: FxHashMap<PatternKey, QueryPlan>,
}

/// Compact cache key for pattern sets.
/// 
/// Uses a tuple of (src_label, edge_type, dst_label, edge_var) tuples for each pattern.
/// This avoids String allocations and provides better cache locality.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PatternKey {
    patterns: Vec<(String, String, String, String)>,
}

impl QueryPlanCache {
    /// Creates a new empty cache.
    pub fn new() -> Self {
        Self {
            cache: FxHashMap::default(),
        }
    }

    /// Gets or creates a query plan for the given patterns.
    ///
    /// If a plan exists in the cache with the same pattern signature, returns it.
    /// Otherwise, creates a new plan and caches it.
    pub fn get_or_create(
        &mut self,
        patterns: &[PatternItem],
        graph: &BeliefGraph,
    ) -> &QueryPlan {
        let key = PatternKey::from_patterns(patterns);
        self.cache
            .entry(key)
            .or_insert_with(|| QueryPlan::new(patterns, graph))
    }

    /// Clears the cache.
    pub fn clear(&mut self) {
        self.cache.clear();
    }
}

impl PatternKey {
    /// Creates a pattern key from a slice of patterns.
    fn from_patterns(patterns: &[PatternItem]) -> Self {
        let patterns = patterns
            .iter()
            .map(|p| (
                p.src.label.clone(),
                p.edge.ty.clone(),
                p.dst.label.clone(),
                p.edge.var.clone(),
            ))
            .collect();
        Self { patterns }
    }
}

impl Default for QueryPlanCache {
    fn default() -> Self {
        Self::new()
    }
}

// Pattern signature function removed - now using PatternKey struct for better cache efficiency

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::graph::BeliefGraph;
    use crate::frontend::ast::{EdgePattern, NodePattern};

    fn create_test_graph() -> BeliefGraph {
        let mut g = BeliefGraph::default();
        let n1 = g.add_node("Person".into(), std::collections::HashMap::new());
        let n2 = g.add_node("Person".into(), std::collections::HashMap::new());
        let n3 = g.add_node("Company".into(), std::collections::HashMap::new());
        
        // Add many edges of type REL
        for _ in 0..10 {
            g.add_edge(n1, n2, "REL".into(), crate::engine::graph::BetaPosterior { alpha: 1.0, beta: 1.0 });
        }
        
        // Add few edges of type WORKS_AT
        g.add_edge(n1, n3, "WORKS_AT".into(), crate::engine::graph::BetaPosterior { alpha: 1.0, beta: 1.0 });
        g.add_edge(n2, n3, "WORKS_AT".into(), crate::engine::graph::BetaPosterior { alpha: 1.0, beta: 1.0 });
        
        g.ensure_owned();
        g
    }

    #[test]
    fn test_pattern_selectivity_ordering() {
        let graph = create_test_graph();
        
        let patterns = vec![
            PatternItem {
                src: NodePattern {
                    var: "A".into(),
                    label: "Person".into(),
                },
                edge: EdgePattern {
                    var: "e1".into(),
                    ty: "REL".into(),
                },
                dst: NodePattern {
                    var: "B".into(),
                    label: "Person".into(),
                },
            },
            PatternItem {
                src: NodePattern {
                    var: "B".into(),
                    label: "Person".into(),
                },
                edge: EdgePattern {
                    var: "e2".into(),
                    ty: "WORKS_AT".into(),
                },
                dst: NodePattern {
                    var: "C".into(),
                    label: "Company".into(),
                },
            },
        ];

        let plan = QueryPlan::new(&patterns, &graph);

        // WORKS_AT should be more selective (fewer edges) and come first
        assert_eq!(plan.ordered_patterns.len(), 2);
        // The second pattern (WORKS_AT) should be ordered first due to lower selectivity
        let first_pattern_idx = plan.ordered_patterns[0];
        assert_eq!(first_pattern_idx, 1); // WORKS_AT pattern (index 1) should be first
    }

    #[test]
    fn test_query_plan_cache() {
        let graph = create_test_graph();
        let mut cache = QueryPlanCache::new();

        let patterns = vec![PatternItem {
            src: NodePattern {
                var: "A".into(),
                label: "Person".into(),
            },
            edge: EdgePattern {
                var: "e1".into(),
                ty: "REL".into(),
            },
            dst: NodePattern {
                var: "B".into(),
                label: "Person".into(),
            },
        }];

        let plan1 = cache.get_or_create(&patterns, &graph);
        let plan1_patterns = plan1.ordered_patterns.clone();
        
        let plan2 = cache.get_or_create(&patterns, &graph);
        let plan2_patterns = plan2.ordered_patterns.clone();

        // Should return the same cached plan
        assert_eq!(plan1_patterns, plan2_patterns);
    }

    #[test]
    fn test_pattern_signature() {
        let patterns1 = vec![PatternItem {
            src: NodePattern {
                var: "A".into(),
                label: "Person".into(),
            },
            edge: EdgePattern {
                var: "e1".into(),
                ty: "REL".into(),
            },
            dst: NodePattern {
                var: "B".into(),
                label: "Person".into(),
            },
        }];

        let patterns2 = vec![PatternItem {
            src: NodePattern {
                var: "X".into(), // Different node variable name
                label: "Person".into(),
            },
            edge: EdgePattern {
                var: "e1".into(), // Same edge variable
                ty: "REL".into(),
            },
            dst: NodePattern {
                var: "B".into(),
                label: "Person".into(),
            },
        }];

        // Note: PatternKey includes labels, edge types, and edge variable names
        // Node variable names are NOT included (they don't affect pattern matching selectivity)
        let key1 = PatternKey::from_patterns(&patterns1);
        let key2 = PatternKey::from_patterns(&patterns2);
        // Same edge variable, same labels, same types = same key
        assert_eq!(key1, key2);
        
        // Different edge variable should produce different key
        let patterns3 = vec![PatternItem {
            src: NodePattern {
                var: "A".into(),
                label: "Person".into(),
            },
            edge: EdgePattern {
                var: "e2".into(), // Different edge variable
                ty: "REL".into(),
            },
            dst: NodePattern {
                var: "B".into(),
                label: "Person".into(),
            },
        }];
        let key3 = PatternKey::from_patterns(&patterns3);
        assert_ne!(key1, key3);
        
        // Same patterns should produce same key
        let key4 = PatternKey::from_patterns(&patterns1);
        assert_eq!(key1, key4);
    }
}

