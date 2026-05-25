use lariat_capability::{CapabilityProfile, Tier};
use lariat_driver::{ContextRef, ContextKind, Turn};
use lariat_physical_model::io::write_step;
use lariat_physical_model::{Step, EnvFeedback, CompileOutcome, TestOutcome};
use lariat_mmie::MmieLoop;
use std::collections::HashMap;
use std::path::Path;

/// Runs one full agent loop: start to completion.
/// This is the TITAN autopilot wire.
pub struct Autopilot<'a> {
    pub driver: &'a lariat_driver::Driver,
    pub inference: &'a lariat_inference::InferenceEngine,
    pub mmie: &'a MmieLoop,
    pub tools: &'a lariat_driver::ToolExecutor,
    pub project_root: &'a Path,
}

impl<'a> Autopilot<'a> {
    pub fn run(&self, request: &str) {
        println!("[titan] Autopilot task: {}\n", request);

        let mut observation = self.build_observation(request);
        let mut step_count = 0usize;
        let mut trajectory_dir: Option<std::path::PathBuf> = None;

        loop {
            step_count += 1;
            if step_count > self.driver.max_steps {
                println!("[titan] Max steps reached. Stopping.");
                break;
            }

            let turn = self.driver.build_turn(request, &observation);
            println!("[titan] Step {} | Mode: {:?}", step_count, turn.mode);

            let context_str = self.format_context(&turn.context_refs);

            let output = match self.inference.generate(&turn.system_prompt,&context_str,&turn.task_query) {
                Ok(text) => text,
                Err(e) => {
                    println!("[titan] Inference error: {}", e);
                    break;
                }
            };

            // Parse and execute any tool calls in the output.
            let (tools_used, tool_results) = self.extract_and_run_tools(&output);

            // Build environment feedback.
            let feedback = EnvFeedback {
                compile_outcome: CompileOutcome::Pass,
                test_outcome: TestOutcome::Skipped,
                logs_summary: tool_results.join("; "),
                memory_used_mb: self.estimate_memory(),
                latency_ms: 100,
            };

            let step = Step {
                mode: format!("{:?}", turn.mode),
                observation_refs: turn.context_refs.into_iter().map(|c| c.path).collect(),
                output,
                tools_used,
                env_feedback: feedback,
                rewards: None,
            };

            // Score with MMIE.
            let rewards = self.mmie.score_step(
                &step,
                step_count,
                12,
                true,
                true,
                None,
            );

            println!("[titan] Reward: {:.2} (compile={:.1} tests={:.1} steps_pen={:.1})",
                rewards.total, rewards.compile, rewards.tests, rewards.steps_penalty
            );

            // Persist step.
            if trajectory_dir.is_none() {
                let root = std::path::PathBuf::from(lariat_physical_model::TRAJECTORIES_DIR);
                let rid = format!("run_{}", uuid::Uuid::new_v4().simple());
                let dir = root.join(&rid);
                let _ = std::fs::create_dir_all(&dir);
                trajectory_dir = Some(dir);
            }
            if let Some(ref dir) = trajectory_dir {
                let name = format!("step_{:03}.json", step_count);
                let _ = write_step(dir, &name, &step);
            }

            // Decide whether to stop.
            if self.driver.should_stop(&step.env_feedback, /* output final-ish */ step.mode == "CODE" && step.output.contains("Done")) {
                println!("[titan] Stop condition met.");
                break;
            }

            if self.driver.needs_rollback(&step.env_feedback) {
                println!("[titan] Too many compile errors; requesting rollback.");
            }

            // Refresh observation for next turn.
            observation = self.build_observation(request);
        }

        println!("\n[titan] Done after {} steps.", step_count);
    }

    fn build_observation(&self, request: &str
    ) -> lariat_driver::Observation {
        use lariat_driver::retriever::{TfIdfRanker, ContextRanker};
        let ranker = TfIdfRanker::new();
        let ranked = ranker.rank(request, self.project_root, &self.driver.profile);
        let mut files: HashMap<lariat_driver::Mode, Vec<String>> = HashMap::new();
        files.insert(lariat_driver::Mode::Code, ranked.iter().map(|r| r.path.clone()).collect());
        lariat_driver::Observation {
            project_tree: vec!["src/main.rs".into(), "src/lib.rs".into()],
            relevant_files: files,
            recent_log: None,
            compiler_errors: vec![],
            test_failures: vec![],
        }
    }

    fn format_context(&self,
        refs: &[ContextRef],
    ) -> String {
        use std::io::Read;
        let mut out = String::new();
        for r in refs.iter() {
            out.push_str(&format!("\n=== {} ===\n", r.path));
            match r.kind {
                ContextKind::Snippet => {
                    if let Ok(mut f) = std::fs::File::open(&r.path) {
                        let mut buf = String::new();
                        let _ = f.read_to_string(&mut buf);
                        // Keep it tiny.
                        let kept = buf.chars().take(800).collect::<String>();
                        out.push_str(&kept);
                        out.push_str("\n<...\n");
                    }
                }
                _ => out.push_str("(summary omitted)\n"),
            }
        }
        out
    }

    fn extract_and_run_tools(
        &self,
        _output: &str,
    ) -> (Vec<lariat_physical_model::ToolCall>, Vec<String>) {
        // In a real system: parse JSON blocks from model output, validate against ToolExecutor, dispatch.
        // For now, emit no tools.
        (vec![], vec!["no tools parsed".into()])
    }

    fn estimate_memory(&self
    ) -> u32 {
        // Rough estimate: model footprint based on tier.
        match self.driver.profile.tier {
            Tier::Ultra => 4000,
            Tier::Standard => 2500,
            Tier::Lite => 1200,
        }
    }
}
