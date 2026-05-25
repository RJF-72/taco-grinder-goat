use lariat_capability::CapabilityProfile;
use anyhow::Result;

pub mod backend;
pub mod llama_cli;
pub use backend::{Backend, GenerationParams};

/// Inference facade over the quantized 6B/1.5B model.
/// Backend is injected; can be MockBackend (testing), LlamaCliBackend (real), or Candle.
pub struct InferenceEngine {
    #[allow(dead_code)]
    profile: CapabilityProfile,
    backend: Box<dyn Backend>,
}

impl InferenceEngine {
    /// Build engine from profile: selects backend variant and quantization tier.
    pub fn from_profile(profile: CapabilityProfile) -> Self {
        let backend = backend::select_backend(&profile);
        Self { profile, backend }
    }

    pub fn from_backend(profile: CapabilityProfile, backend: Box<dyn Backend>) -> Self {
        Self { profile, backend }
    }

    pub fn backend(&self) -> &dyn Backend {
        self.backend.as_ref()
    }

    /// Run one generation turn.
    /// Input: driver Turn (system prompt + context + task)
    /// Output: model message (plan, tools, code, explain)
    pub fn generate(
        &self,
        system_prompt: &str,
        context: &str,
        task: &str,
    ) -> Result<String> {
        let prompt = format!(
            "<|system|>\n{}\n<|context|>\n{}\n<|task|>\n{}",
            system_prompt, context, task
        );
        let params = GenerationParams {
            max_tokens: 2048,
            temperature: 0.2,
            stop: vec!["<|end|>".into()],
        };
        self.backend.generate(&prompt, &params)
    }
}
