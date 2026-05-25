// LARIAT MMIE
// Multi-Modal Incremental Enrichment loop.
// Reasoning + self-improvement. Scores trajectories and curates training data.

use lariat_physical_model::{Rewards, Step, TrajectorySummary};
use lariat_capability::{CapabilityProfile, Tier};

pub struct MmieLoop {
    profile: CapabilityProfile,
    // Thresholds tuned per-tier.
    thresholds: RewardThresholds,
}

#[derive(Debug, Clone)]
pub struct RewardThresholds {
    pub min_total_to_keep: f64,
    pub max_latency_ms: u32,
    pub max_memory_mb: u32,
}

impl Default for RewardThresholds {
    fn default() -> Self {
        Self {
            min_total_to_keep: 0.6,
            max_latency_ms: 10_000,
            max_memory_mb: 5_000,
        }
    }
}

impl MmieLoop {
    pub fn new(profile: CapabilityProfile) -> Self {
        Self {
            thresholds: RewardThresholds::default(),
            profile,
        }
    }

    /// Score a single step using environment feedback and driver history.
    /// This is the reasoning engine.
    pub fn score_step(
        &self,
        step: &Step,
        num_steps: usize,
        diff_lines: usize,
        requirement_matched: bool,
        explanation_matches_change: bool,
        vision_correct: Option<bool>,
    ) -> Rewards {
        // --- compile ---
        let compile_r = match &step.env_feedback.compile_outcome {
            lariat_physical_model::CompileOutcome::Pass => 1.0,
            lariat_physical_model::CompileOutcome::Fail(errs) => {
                if errs.len() <= 1 {
                    0.4
                } else {
                    0.0
                }
            }
        };

        // --- tests ---
        let tests_r = match &step.env_feedback.test_outcome {
            lariat_physical_model::TestOutcome::Pass => 1.0,
            lariat_physical_model::TestOutcome::Fail(errs) => {
                if errs.len() <= 2 { 0.3 } else { 0.0 }
            }
            lariat_physical_model::TestOutcome::Skipped => 0.0,
        };

        // step penalty: fewer steps = better; penalize by count
        let steps_penalty = -(num_steps.saturating_sub(3) as f64 * 0.1);

        // diff size: encourage minimal, sufficient changes
        let diff_penalty = if diff_lines > 50 {
            -(diff_lines.saturating_sub(50) as f64 * 0.005)
        } else {
            0.0
        };

        // tool correctness
        let tools_r = 1.0; // stub; can be improved with schema validation

        let alignment_r = if requirement_matched { 1.0 } else { 0.0 };
        let explain_r = if explanation_matches_change { 1.0 } else { 0.0 };
        let vision_r = vision_correct.map(|b| if b { 1.0 } else { 0.0 });

        // memory/latency penalties
        let mem_penalty = if step.env_feedback.memory_used_mb > self.thresholds.max_memory_mb {
            -(step.env_feedback.memory_used_mb.saturating_sub(self.thresholds.max_memory_mb) as f64 * 0.0005)
        } else {
            0.0
        };
        let lat_penalty = if step.env_feedback.latency_ms > self.thresholds.max_latency_ms {
            -(step.env_feedback.latency_ms.saturating_sub(self.thresholds.max_latency_ms) as f64 * 0.0001)
        } else {
            0.0
        };

        let total = compile_r
            + tests_r
            + steps_penalty
            + diff_penalty
            + tools_r
            + alignment_r
            + explain_r
            + vision_r.unwrap_or(0.0)
            + mem_penalty
            + lat_penalty;

        let total = total.clamp(-5.0, 5.0);

        Rewards {
            compile: compile_r,
            tests: tests_r,
            steps_penalty,
            diff_size_penalty: diff_penalty,
            tools: tools_r,
            alignment: alignment_r,
            explain_quality: explain_r,
            vision: vision_r,
            memory_penalty: mem_penalty,
            latency_penalty: lat_penalty,
            total,
        }
    }

    /// MMIE runs this after a trajectory is complete.
    /// It returns true if this trajectory is good enough for retraining.
    pub fn accept_trajectory(&self, summary: &TrajectorySummary) -> bool {
        if !summary.success {
            return summary.final_reward >= self.thresholds.min_total_to_keep + 0.2;
        }
        summary.final_reward >= self.thresholds.min_total_to_keep
    }

    /// Adjust thresholds based on tier.
    pub fn tune_for_tier(&mut self) {
        self.thresholds.max_latency_ms = match self.profile.tier {
            Tier::Ultra => 10_000,
            Tier::Standard => 7_000,
            Tier::Lite => 5_000,
        };
        self.thresholds.max_memory_mb = self.profile.limits.max_model_mem_mb;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lariat_physical_model::{CompileOutcome, TestOutcome, EnvFeedback};
    use lariat_capability::CapabilityProfile;

    fn dummy_env() -> EnvFeedback {
        EnvFeedback {
            compile_outcome: CompileOutcome::Pass,
            test_outcome: TestOutcome::Pass,
            logs_summary: "ok".into(),
            memory_used_mb: 500,
            latency_ms: 800,
        }
    }

    #[test]
    fn good_step_scores_high() {
        let prof = CapabilityProfile::probe();
        let mmie = MmieLoop::new(prof);
        let s = lariat_physical_model::Step {
            mode: "CODE".into(),
            observation_refs: vec![],
            output: "fn main() {}".into(),
            tools_used: vec![],
            env_feedback: dummy_env(),
            rewards: None,
        };
        let r = mmie.score_step(&s, 3, 12, true, true, None);
        assert!(r.total >= 3.0, "expected high total, got {}", r.total);
    }
}
