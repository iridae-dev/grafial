//! Parallel flow execution with pipeline parallelism.
//!
//! This module enables parallel execution of flow pipelines, where independent
//! graph transformations and computations can run concurrently.
//!
//! ## Architecture
//!
//! - **Pipeline stages**: Flows decomposed into parallelizable stages
//! - **Data parallelism**: Multiple graphs processed in parallel
//! - **Task parallelism**: Different transforms applied concurrently
//! - **Stream processing**: Continuous flow execution with backpressure
//!
//! ## Feature gating
//!
//! Parallel flow execution is behind the `parallel` feature flag.

use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use crate::engine::errors::ExecError;
use crate::engine::graph::BeliefGraph;
use grafial_ir::{FlowIR, GraphDefIR, TransformIR};

/// Parallel flow executor with pipeline and data parallelism.
#[cfg(feature = "parallel")]
pub struct ParallelFlowExecutor {
    /// Worker threads for parallel execution
    #[allow(dead_code)]
    workers: Vec<FlowWorker>,
    /// Scheduler for distributing work
    scheduler: Arc<FlowScheduler>,
    /// Pipeline configuration
    config: FlowConfig,
}

/// Configuration for parallel flow execution.
#[derive(Debug, Clone)]
pub struct FlowConfig {
    /// Number of worker threads
    pub num_workers: usize,
    /// Size of the work queue per worker
    pub queue_size: usize,
    /// Enable pipeline parallelism
    pub pipeline_parallel: bool,
    /// Enable data parallelism
    pub data_parallel: bool,
    /// Maximum graphs to process in parallel
    pub max_parallel_graphs: usize,
}

impl Default for FlowConfig {
    fn default() -> Self {
        Self {
            num_workers: num_cpus::get(),
            queue_size: 100,
            pipeline_parallel: true,
            data_parallel: true,
            max_parallel_graphs: 4,
        }
    }
}

/// A worker thread for flow execution.
#[allow(dead_code)]
struct FlowWorker {
    /// Worker ID
    id: usize,
    /// Receiving channel for work
    receiver: Receiver<FlowTask>,
    /// Sending channel for results
    sender: Sender<FlowResult>,
}

/// Scheduler for distributing flow tasks to workers.
struct FlowScheduler {
    /// Task queue per worker
    worker_queues: Vec<Sender<FlowTask>>,
    /// Result aggregator
    result_receiver: Receiver<FlowResult>,
    /// Current worker for round-robin
    current_worker: RwLock<usize>,
    /// Statistics
    stats: RwLock<SchedulerStats>,
}

/// A task to be executed by a worker.
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum FlowTask {
    /// Apply a transform to a graph
    Transform {
        graph_id: String,
        graph: Arc<RwLock<BeliefGraph>>,
        transform: TransformIR,
    },
    /// Evaluate a graph expression
    EvaluateGraph { graph_id: String, expr: GraphDefIR },
    /// Compute metrics on a graph
    ComputeMetrics {
        graph_id: String,
        graph: Arc<RwLock<BeliefGraph>>,
        metric_names: Vec<String>,
    },
    /// Synchronization barrier
    Barrier { barrier_id: usize },
}

/// Result of a flow task execution.
#[derive(Debug)]
#[allow(dead_code)]
enum FlowResult {
    /// Transform completed
    TransformComplete { graph_id: String, transform: String },
    /// Graph evaluation completed
    GraphComplete {
        graph_id: String,
        graph: Arc<RwLock<BeliefGraph>>,
    },
    /// Metrics computed
    MetricsComplete {
        graph_id: String,
        values: HashMap<String, f64>,
    },
    /// Barrier reached
    BarrierReached { barrier_id: usize, worker_id: usize },
    /// Error occurred
    Error { task: String, error: ExecError },
}

/// Statistics about scheduler performance.
#[derive(Debug, Default)]
#[allow(dead_code)]
struct SchedulerStats {
    /// Total tasks scheduled
    tasks_scheduled: usize,
    /// Total tasks completed
    tasks_completed: usize,
    /// Tasks per worker
    tasks_per_worker: HashMap<usize, usize>,
    /// Average task latency in ms
    avg_latency_ms: f64,
}

#[cfg(feature = "parallel")]
impl ParallelFlowExecutor {
    /// Create a new parallel flow executor.
    pub fn new(config: FlowConfig) -> Self {
        let (result_sender, result_receiver) = unbounded();
        let mut worker_queues = Vec::new();
        let mut workers = Vec::new();

        // Create workers
        for id in 0..config.num_workers {
            let (task_sender, task_receiver) = bounded(config.queue_size);
            worker_queues.push(task_sender);

            workers.push(FlowWorker {
                id,
                receiver: task_receiver,
                sender: result_sender.clone(),
            });
        }

        let scheduler = Arc::new(FlowScheduler {
            worker_queues,
            result_receiver,
            current_worker: RwLock::new(0),
            stats: RwLock::new(SchedulerStats::default()),
        });

        Self {
            workers,
            scheduler,
            config,
        }
    }

    /// Execute a flow in parallel.
    pub fn execute_flow(&self, flow: &FlowIR) -> Result<FlowExecutionResult, ExecError> {
        let execution_plan = self.build_execution_plan(flow)?;
        let mut results = FlowExecutionResult::default();

        // Execute stages in order
        for stage in &execution_plan.stages {
            match stage {
                ExecutionStage::Parallel(tasks) => {
                    self.execute_parallel_tasks(tasks, &mut results)?;
                }
                ExecutionStage::Sequential(task) => {
                    self.execute_sequential_task(task, &mut results)?;
                }
                ExecutionStage::Barrier(id) => {
                    self.synchronize_workers(*id)?;
                }
            }
        }

        Ok(results)
    }

    /// Build an execution plan for a flow.
    fn build_execution_plan(&self, flow: &FlowIR) -> Result<ExecutionPlan, ExecError> {
        let mut stages = Vec::new();

        // Analyze graph dependencies
        let graph_deps = self.analyze_graph_dependencies(flow)?;

        // Group independent graphs for parallel execution
        for level in graph_deps.levels {
            if level.len() > 1 && self.config.data_parallel {
                // Multiple independent graphs - execute in parallel
                let tasks: Vec<_> = level
                    .into_iter()
                    .map(|graph_name| {
                        let graph_def = flow
                            .graphs
                            .iter()
                            .find(|g| g.name == graph_name)
                            .ok_or_else(|| {
                                ExecError::ValidationError(format!(
                                    "Graph {} not found",
                                    graph_name
                                ))
                            })?;

                        Ok(FlowTask::EvaluateGraph {
                            graph_id: graph_name,
                            expr: graph_def.clone(),
                        })
                    })
                    .collect::<Result<Vec<_>, ExecError>>()?;

                stages.push(ExecutionStage::Parallel(tasks));
            } else {
                // Single graph or sequential mode
                for graph_name in level {
                    let graph_def = flow
                        .graphs
                        .iter()
                        .find(|g| g.name == graph_name)
                        .ok_or_else(|| {
                            ExecError::ValidationError(format!("Graph {} not found", graph_name))
                        })?;

                    stages.push(ExecutionStage::Sequential(FlowTask::EvaluateGraph {
                        graph_id: graph_name,
                        expr: graph_def.clone(),
                    }));
                }
            }

            // Add barrier after each level
            stages.push(ExecutionStage::Barrier(stages.len()));
        }

        Ok(ExecutionPlan { stages })
    }

    /// Analyze dependencies between graphs in a flow.
    fn analyze_graph_dependencies(&self, flow: &FlowIR) -> Result<GraphDependencies, ExecError> {
        let mut deps = HashMap::new();

        for graph in &flow.graphs {
            let graph_deps = self.extract_graph_dependencies(graph);
            deps.insert(graph.name.clone(), graph_deps);
        }

        // Compute topological levels
        let levels = self.compute_dependency_levels(&deps)?;

        Ok(GraphDependencies { deps, levels })
    }

    /// Extract dependencies from a graph definition.
    fn extract_graph_dependencies(&self, graph_def: &GraphDefIR) -> HashSet<String> {
        let mut deps = HashSet::new();

        // Extract dependencies from the graph expression
        match &graph_def.expr {
            grafial_ir::GraphExprIR::FromEvidence(_) => {
                // No graph dependencies for evidence-based graphs
            }
            grafial_ir::GraphExprIR::FromGraph(alias) => {
                // Depends on the imported graph
                deps.insert(alias.clone());
            }
            grafial_ir::GraphExprIR::Pipeline { start_graph, .. } => {
                // Depends on the starting graph
                deps.insert(start_graph.clone());
            }
        }

        deps
    }

    /// Compute dependency levels for parallel execution.
    fn compute_dependency_levels(
        &self,
        deps: &HashMap<String, HashSet<String>>,
    ) -> Result<Vec<Vec<String>>, ExecError> {
        // Simplified topological sort
        let mut levels = Vec::new();
        let mut remaining: HashSet<_> = deps.keys().cloned().collect();
        let mut processed = HashSet::new();

        while !remaining.is_empty() {
            let mut current_level = Vec::new();

            for graph_name in remaining.clone() {
                let graph_deps = &deps[&graph_name];
                // Check if all dependencies have been processed
                if graph_deps.iter().all(|dep| processed.contains(dep)) {
                    current_level.push(graph_name.clone());
                }
            }

            if current_level.is_empty() {
                return Err(ExecError::ValidationError(
                    "Circular dependency in flow graphs".to_string(),
                ));
            }

            // Move items from current level to processed
            for graph_name in &current_level {
                remaining.remove(graph_name);
                processed.insert(graph_name.clone());
            }

            levels.push(current_level);
        }

        Ok(levels)
    }

    /// Execute tasks in parallel.
    fn execute_parallel_tasks(
        &self,
        tasks: &[FlowTask],
        results: &mut FlowExecutionResult,
    ) -> Result<(), ExecError> {
        // Distribute tasks to workers
        for task in tasks {
            self.scheduler.schedule_task(task.clone())?;
        }

        // Collect results
        for _ in tasks {
            match self.scheduler.wait_for_result() {
                Ok(result) => self.process_result(result, results)?,
                Err(e) => return Err(e),
            }
        }

        Ok(())
    }

    /// Execute a single task sequentially.
    fn execute_sequential_task(
        &self,
        task: &FlowTask,
        results: &mut FlowExecutionResult,
    ) -> Result<(), ExecError> {
        self.scheduler.schedule_task(task.clone())?;

        match self.scheduler.wait_for_result() {
            Ok(result) => self.process_result(result, results),
            Err(e) => Err(e),
        }
    }

    /// Synchronize all workers at a barrier.
    fn synchronize_workers(&self, barrier_id: usize) -> Result<(), ExecError> {
        // Send barrier to all workers
        for _ in 0..self.config.num_workers {
            self.scheduler
                .schedule_task(FlowTask::Barrier { barrier_id })?;
        }

        // Wait for all workers to reach barrier
        for _ in 0..self.config.num_workers {
            match self.scheduler.wait_for_result() {
                Ok(FlowResult::BarrierReached { .. }) => {}
                Ok(_) => {
                    return Err(ExecError::Internal(
                        "Unexpected result at barrier".to_string(),
                    ))
                }
                Err(e) => return Err(e),
            }
        }

        Ok(())
    }

    /// Process a result from a worker.
    fn process_result(
        &self,
        result: FlowResult,
        results: &mut FlowExecutionResult,
    ) -> Result<(), ExecError> {
        match result {
            FlowResult::GraphComplete { graph_id, graph } => {
                results.graphs.insert(graph_id, graph);
            }
            FlowResult::MetricsComplete { graph_id, values } => {
                results.metrics.insert(graph_id, values);
            }
            FlowResult::Error { task, error } => {
                return Err(ExecError::Internal(format!(
                    "Task {} failed: {}",
                    task, error
                )));
            }
            _ => {}
        }

        Ok(())
    }
}

impl FlowScheduler {
    /// Schedule a task to a worker.
    fn schedule_task(&self, task: FlowTask) -> Result<(), ExecError> {
        // Round-robin scheduling
        let worker_id = {
            let mut current = self.current_worker.write().unwrap();
            let id = *current;
            *current = (*current + 1) % self.worker_queues.len();
            id
        };

        // Send to worker
        self.worker_queues[worker_id]
            .send(task)
            .map_err(|_| ExecError::Internal("Worker queue full".to_string()))?;

        // Update stats
        let mut stats = self.stats.write().unwrap();
        stats.tasks_scheduled += 1;
        *stats.tasks_per_worker.entry(worker_id).or_insert(0) += 1;

        Ok(())
    }

    /// Wait for a result from any worker.
    fn wait_for_result(&self) -> Result<FlowResult, ExecError> {
        self.result_receiver
            .recv()
            .map_err(|_| ExecError::Internal("Result channel closed".to_string()))
    }
}

/// Execution plan for a flow.
struct ExecutionPlan {
    stages: Vec<ExecutionStage>,
}

/// A stage in the execution plan.
enum ExecutionStage {
    /// Tasks that can run in parallel
    Parallel(Vec<FlowTask>),
    /// Task that must run alone
    Sequential(FlowTask),
    /// Synchronization barrier
    Barrier(usize),
}

/// Dependencies between graphs.
struct GraphDependencies {
    /// Graph name -> dependencies
    #[allow(dead_code)]
    deps: HashMap<String, HashSet<String>>,
    /// Topological levels
    levels: Vec<Vec<String>>,
}

/// Result of flow execution.
#[derive(Debug, Default)]
pub struct FlowExecutionResult {
    /// Resulting graphs
    pub graphs: HashMap<String, Arc<RwLock<BeliefGraph>>>,
    /// Computed metrics
    pub metrics: HashMap<String, HashMap<String, f64>>,
    /// Execution statistics
    pub stats: ExecutionStats,
}

/// Statistics about flow execution.
#[derive(Debug, Default)]
pub struct ExecutionStats {
    /// Total execution time in ms
    pub total_time_ms: u64,
    /// Time per stage in ms
    pub stage_times_ms: Vec<u64>,
    /// Parallelism achieved
    pub avg_parallelism: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_flow_config() {
        let config = FlowConfig::default();
        assert!(config.num_workers > 0);
        assert!(config.pipeline_parallel);
        assert!(config.data_parallel);
    }

    #[test]
    fn test_dependency_levels() {
        let executor = ParallelFlowExecutor::new(FlowConfig::default());

        let mut deps = HashMap::new();
        deps.insert("g1".to_string(), HashSet::new());
        deps.insert("g2".to_string(), HashSet::new());
        deps.insert("g3".to_string(), {
            let mut s = HashSet::new();
            s.insert("g1".to_string());
            s.insert("g2".to_string());
            s
        });

        let levels = executor.compute_dependency_levels(&deps).unwrap();
        assert_eq!(levels.len(), 2);
        assert_eq!(levels[0].len(), 2); // g1 and g2
        assert_eq!(levels[1].len(), 1); // g3
    }
}
