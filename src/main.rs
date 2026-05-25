// TITAN — Top-level integration example
// Wires Driver, Inference, MMIE, and Physical Model into one loop.

use lariat_capability::CapabilityProfile;
use lariat_driver::Driver;
use lariat_inference::InferenceEngine;
use lariat_mmie::MmieLoop;
use std::path::Path;

fn main() {
    println!("=== TITAN bootstrap ===");
    // 1. Probe machine
    let profile = CapabilityProfile::probe();
    println!("Profile: {:?} | tier: {:?}", profile, profile.tier);

    // 2. Construct driver with memory-aware defaults
    let driver = Driver::new(profile.clone(), ".");

    // 3. Inference engine (MockBackend by default unless .gguf model present)
    let inference = InferenceEngine::from_profile(profile.clone());

    // 4. MMIE reasoning loop
    let mmie = MmieLoop::new(profile.clone());

    // 5. Tool executor
    let tools = lariat_driver::ToolExecutor::with_defaults();

    // 6. Start task
    let request = "Refactor the auth middleware to use async closures and add tracing spans.";
    println!("Task: {}", request);

    // 7. Run autopilot turn
    let turn = driver.build_turn(request, &build_observation(request, &profile));
    println!("Mode: {:?}", turn.mode);

    let context_str = format_context(&turn.context_refs);

    let output = inference
        .generate(&turn.system_prompt, &context_str, &turn.task_query)
        .expect("inference failed");
    println!("Model output:\n{}", output);

    // 8. Parse tool calls from output and validate
    let mut tool_results: Vec<String> = Vec::new();
    if let Ok(tc) = serde_json::from_str::<lariat_physical_model::ToolCall>(&output) {
        println!("Parsed tool call: {:?}", tc.tool);
        let _ = tools.validate(&tc.tool, &tc.args);
        let result = lariat_driver::tools::dispatch(&tc.tool, &tc.args);
        match result {
            Ok(text) => { println!("Tool result: {}", text); tool_results.push(text); }
            Err(e) => println!("Tool dispatch error: {}", e),
        }
    }

    // 9. Build environment feedback
    let feedback = lariat_physical_model::EnvFeedback {
        compile_outcome: lariat_physical_model::CompileOutcome::Pass,
        test_outcome: lariat_physical_model::TestOutcome::Skipped,
        logs_summary: tool_results.join("; "),
        memory_used_mb: estimate_memory(&profile.tier),
        latency_ms: 1200,
    };

    // 10. Build and score step
    let step = lariat_physical_model::Step {
        mode: format!("{:?}", turn.mode),
        observation_refs: turn
            .context_refs
            .into_iter()
            .map(|c| c.path)
            .collect(),
        output,
        tools_used: vec![],
        env_feedback: feedback,
        rewards: None,
    };

    let rewards = mmie.score_step(&step, 1, 18, true, true, None);
    println!("MMIE total reward: {:.2}", rewards.total);

    println!("\n=== Done ===");
}

fn build_observation(request: &str, profile: &CapabilityProfile) -> lariat_driver::Observation {
    use lariat_driver::retriever::{ContextRanker, TfIdfRanker};
    let ranker = TfIdfRanker::new();
    let ranked = ranker.rank(request, Path::new("."), profile);
    let mut files = std::collections::HashMap::new();
    files.insert(
        lariat_driver::Mode::Code,
        ranked.into_iter().map(|r| r.path).collect(),
    );
    lariat_driver::Observation {
        project_tree: vec!["src/main.rs".into(), "src/lib.rs".into()],
        relevant_files: files,
        recent_log: None,
        compiler_errors: vec![],
        test_failures: vec![],
    }
}

fn format_context(refs: &[lariat_driver::ContextRef]) -> String {
    use std::io::Read;
    let mut out = String::new();
    for r in refs.iter() {
        out.push_str(&format!("\n=== {} ===\n", r.path));
        match r.kind {
            lariat_driver::ContextKind::Snippet => {
                if let Ok(mut f) = std::fs::File::open(&r.path) {
                    let mut buf = String::new();
                    let _ = f.read_to_string(&mut buf);
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

fn estimate_memory(tier: &lariat_capability::Tier) -> u32 {
    match tier {
        lariat_capability::Tier::Ultra => 4000,
        lariat_capability::Tier::Standard => 2500,
        lariat_capability::Tier::Lite => 1200,
    }
}
