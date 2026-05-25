use anyhow::Result;
use lariat_capability::CapabilityProfile;

/// Backend contract for model generation.
pub trait Backend: Send + Sync {
    /// Generate text from a prompt.
    fn generate(&self,
        prompt: &str,
        params: &GenerationParams,
    ) -> Result<String>;
}

/// Sampling parameters forwarded to the underlying engine.
#[derive(Debug, Clone)]
pub struct GenerationParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub stop: Vec<String>,
}

/// Mock backend for testing: returns a plausible-looking stub.
pub struct MockBackend;

impl Backend for MockBackend {
    fn generate(
        &self,
        _prompt: &str,
        _params: &GenerationParams,
    ) -> Result<String> {
        Ok(format!(
            "I will refactor the auth middleware.\n{{\"tool\":\"open_file\",\"args\":{{\"path\":\"src/auth.rs\"}}}}\n<|end|>"
        ))
    }
}

/// Select a backend based on capability profile.
/// Prefers LlamaCliBackend if a model file is present; otherwise falls back to MockBackend.
pub fn select_backend(profile: &CapabilityProfile) -> Box<dyn Backend> {
    if let Some(cli) = crate::llama_cli::LlamaCliBackend::discover(profile) {
        println!("[titan] LlamaCliBackend active: model={:?}", cli.model_path);
        return Box::new(cli);
    }
    println!("[titan] No llama-cli or GGUF model found; using MockBackend.");
    Box::new(MockBackend)
}
