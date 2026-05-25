// LARIAT DRIVER
// Deterministic, schema-driven, mode-based agent conductor.
// Perfect = deterministic, memory-aware, tool-strict, tier-aware, mode-explicit.

use lariat_capability::{CapabilityProfile, Tier};
use lariat_physical_model::EnvFeedback;
pub mod retriever;
pub mod tools;
pub use tools::{ToolExecutor, ToolError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Modes the driver routes into. Never blended—always explicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Mode {
    Requirement,
    Spec,
    Code,
    Explain,
    Vision,
}

/// A Turn is the atomic unit of the agent loop.
/// Driver decides Mode -> builds prompt -> calls inference -> parses output -> tools.
#[derive(Debug, Clone)]
pub struct Turn {
    pub mode: Mode,
    pub system_prompt: String,
    pub context_refs: Vec<ContextRef>,
    pub task_query: String,
    pub previous_summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextRef {
    pub path: String,
    // Summary or raw snippet?
    pub kind: ContextKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContextKind {
    Full,
    Summary,
    Snippet,
    LogTail,
}

/// Driver state for the autopilot loop.
pub struct Driver {
    pub profile: CapabilityProfile,
    pub mode: Mode,
    pub step_count: usize,
    pub max_steps: usize,
    pub pending_rollback: Option<String>,
    pub project_root: String,
}

impl Driver {
    pub fn new(profile: CapabilityProfile, project_root: impl Into<String>) -> Self {
        let max_steps = match profile.tier {
            Tier::Ultra => 20,
            Tier::Standard => 15,
            Tier::Lite => 10,
        };
        Self {
            profile,
            mode: Mode::Requirement,
            step_count: 0,
            max_steps,
            pending_rollback: None,
            project_root: project_root.into(),
        }
    }

    /// Route a raw user request into an initial Mode.
    /// In a full system, an ML classifier may refine this; for now, deterministic.
    pub fn classify(&self, request: &str) -> Mode {
        if request.starts_with("explain") || request.starts_with("summarize") || request.starts_with("doc") {
            Mode::Explain
        } else if request.starts_with("read this spec") || request.starts_with("rfc") || request.starts_with("spec") {
            Mode::Spec
        } else if request.starts_with("vision") || request.starts_with("screenshot") || request.starts_with("diagram") {
            if self.profile.features.vision_enabled {
                Mode::Vision
            } else {
                Mode::Code
            }
        } else {
            Mode::Code
        }
    }

    /// Build the Turn from the current state.
    pub fn build_turn(&self, request: &str, observation: &Observation) -> Turn {
        let mode = self.classify(request);
        let sys = self.system_prompt_for(&mode);
        let ctx = self.select_context(observation);
        let prev = format!("{} prior steps completed.", self.step_count);

        Turn {
            mode,
            system_prompt: sys,
            context_refs: ctx,
            task_query: request.into(),
            previous_summary: prev,
        }
    }

    fn system_prompt_for(&self, mode: &Mode) -> String {
        let base = "You are a Rust engineer inside Lariat. You have access to a set of tools and you must obey the schema. Never invent files. Always run tests before finalizing.";
        let tier_note = format!("Current tier: {:?}. Memory budget: {}MB, context budget: {} tokens.",
             self.profile.tier, self.profile.limits.max_model_mem_mb, self.profile.limits.max_context_tokens);
        let mode_contract = match mode {
            Mode::Requirement => "MODE: REQUIREMENT. Read the user's request, produce a structured plan with steps, files to change, and constraints.",
            Mode::Spec => "MODE: SPEC. Read the provided spec/RFC/README and extract constraints, invariants, and APIs in structured form.",
            Mode::Code => "MODE: CODE. Given the plan and context, produce a minimal patch, call tools, and reason step-by-step.",
            Mode::Explain => "MODE: EXPLAIN. Summarize what changed and why in plain English. Keep it tight.",
            Mode::Vision => "MODE: VISION. Extract structure and intent from the provided screenshot/diagram.",
        };
        format!("{}\n{}\n{}", base, tier_note, mode_contract)
    }

    fn select_context(&self, obs: &Observation) -> Vec<ContextRef> {
        let mut out = Vec::new();
        // Never load entire repo. Pick top-K relevant files.
        // In a real system, an embedding retriever ranks paths.
        for f in obs.relevant_files.get(&self.mode).unwrap_or(&vec![]).iter() {
            out.push(ContextRef { path: f.clone(), kind: ContextKind::Snippet });
        }
        // Aggressively summarize large logs.
        if let Some(log) = &obs.recent_log {
            out.push(ContextRef { path: log.clone(), kind: ContextKind::LogTail });
        }
        out
    }

    /// Should the loop stop? Guard rail.
    pub fn should_stop(&self, last_feedback: &EnvFeedback, output_looks_final: bool) -> bool {
        if self.step_count >= self.max_steps {
            return true;
        }
        if matches!(last_feedback.compile_outcome, lariat_physical_model::CompileOutcome::Pass)
            && matches!(last_feedback.test_outcome, lariat_physical_model::TestOutcome::Pass)
            && output_looks_final {
            return true;
        }
        false
    }

    /// Determine whether to rollback the last patch after a critical failure.
    pub fn needs_rollback(&self, last_feedback: &EnvFeedback) -> bool {
        match &last_feedback.compile_outcome {
            lariat_physical_model::CompileOutcome::Fail(errs) if errs.len() > 3 => true,
            _ => false,
        }
    }
}

/// Environment observation captured before each turn.
pub struct Observation {
    pub project_tree: Vec<String>,
    pub relevant_files: HashMap<Mode, Vec<String>>,
    pub recent_log: Option<String>,
    pub compiler_errors: Vec<String>,
    pub test_failures: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn driver_routes_explain_mode() {
        let prof = CapabilityProfile::probe();
        let d = Driver::new(prof, ".");
        assert_eq!(d.classify("explain this error"), Mode::Explain);
    }

    #[test]
    fn driver_limits_steps() {
        let prof = CapabilityProfile::probe();
        let d = Driver::new(prof, ".");
        assert!(d.max_steps > 0);
    }
}
