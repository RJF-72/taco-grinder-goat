// LARIAT PHYSICAL MODEL
// On-disk structured memory: trajectories, diffs, errors, fixes, rewards.
// Never shove this into RAM; stream from disk, summarize aggressively.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const TRAJECTORIES_DIR: &str = ".lariat/training/trajectories";
pub const DIFFS_DIR: &str = ".lariat/training/diffs";
pub const ERRORS_DIR: &str = ".lariat/training/errors";
pub const REWARDS_DIR: &str = ".lariat/training/rewards";

/// One step in a trajectory.
/// Recorded by the driver at every turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step {
    pub mode: String,
    pub observation_refs: Vec<String>,
    pub output: String,
    pub tools_used: Vec<ToolCall>,
    pub env_feedback: EnvFeedback,
    // Filled by MMIE later.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rewards: Option<Rewards>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub tool: String,
    pub args: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvFeedback {
    pub compile_outcome: CompileOutcome,
    pub test_outcome: TestOutcome,
    pub logs_summary: String,
    pub memory_used_mb: u32,
    pub latency_ms: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompileOutcome {
    Pass,
    Fail(Vec<String>),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TestOutcome {
    Pass,
    Fail(Vec<String>),
    Skipped,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rewards {
    pub compile: f64,
    pub tests: f64,
    pub steps_penalty: f64,
    pub diff_size_penalty: f64,
    pub tools: f64,
    pub alignment: f64,
    pub explain_quality: f64,
    pub vision: Option<f64>,
    pub memory_penalty: f64,
    pub latency_penalty: f64,
    pub total: f64,
}

/// A trajectory is a full run from task start to completion.
/// Stored on disk as a directory with step files + a summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectorySummary {
    pub run_id: uuid::Uuid,
    pub tier: String,
    pub tags: Vec<String>,
    pub success: bool,
    pub num_steps: usize,
    pub final_reward: f64,
    pub model_variant: String,
    pub driver_version: String,
}

pub mod io;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_roundtrip() {
        let s = Step {
            mode: "CODE".into(),
            observation_refs: vec!["file.rs".into()],
            output: "fn main() {}".into(),
            tools_used: vec![],
            env_feedback: EnvFeedback {
                compile_outcome: CompileOutcome::Pass,
                test_outcome: TestOutcome::Skipped,
                logs_summary: "ok".into(),
                memory_used_mb: 300,
                latency_ms: 200,
            },
            rewards: None,
        };
        let json = serde_json::to_string(&s).unwrap();
        let back: Step = serde_json::from_str(&json).unwrap();
        assert_eq!(back.mode, "CODE");
    }
}
