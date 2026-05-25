use anyhow::Result;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use lariat_capability::CapabilityProfile;
use crate::backend::{Backend, GenerationParams};

/// Llama.cpp CLI backend: shells out to `llama-cli` (or `llama-cli.exe` on Windows).
/// Keeps the Rust workspace C++-free.
pub struct LlamaCliBackend {
    pub cli_path: PathBuf,
    pub model_path: PathBuf,
    pub ctx_size: usize,
    pub n_threads: usize,
}

impl LlamaCliBackend {
    /// Discover llama-cli and a matching GGUF model.
    pub fn discover(profile: &CapabilityProfile) -> Option<Self> {
        let cli = Self::find_llama_cli()?;
        let model = Self::find_best_model(profile)?;
        let n_threads = profile.cpu_cores as usize;
        Some(Self {
            cli_path: cli,
            model_path: model,
            ctx_size: profile.limits.max_context_tokens as usize,
            n_threads,
        })
    }

    fn find_llama_cli() -> Option<PathBuf> {
        let bundled = if cfg!(windows) {
            PathBuf::from(".lariat/bin/llama-cli.exe")
        } else {
            PathBuf::from(".lariat/bin/llama-cli")
        };
        if bundled.exists() { return Some(bundled); }
        let name = if cfg!(windows) { "llama-cli.exe" } else { "llama-cli" };
        if let Ok(paths) = std::env::var("PATH") {
            for dir in paths.split(if cfg!(windows) { ';' } else { ':' }) {
                let candidate = PathBuf::from(dir).join(name);
                if candidate.exists() { return Some(candidate); }
            }
        }
        None
    }

    fn find_best_model(profile: &CapabilityProfile) -> Option<PathBuf> {
        let models_dir = PathBuf::from(".lariat/models");
        if !models_dir.exists() { return None; }
        let mut candidates: Vec<PathBuf> = Vec::new();
        for entry in std::fs::read_dir(&models_dir).ok()? {
            let entry = entry.ok()?;
            let path = entry.path();
            if let Some(ext) = path.extension() {
                if ext.eq_ignore_ascii_case("gguf") {
                    candidates.push(path);
                }
            }
        }
        if candidates.is_empty() { return None; }
        candidates.sort_by_key(|p| {
            std::fs::metadata(p)
                .map(|m| m.len())
                .unwrap_or(u64::MAX)
        });
        match profile.tier {
            lariat_capability::Tier::Lite => candidates.first().cloned(),
            lariat_capability::Tier::Standard => {
                let idx = candidates.len() / 2;
                Some(candidates[idx].clone())
            }
            lariat_capability::Tier::Ultra => candidates.last().cloned(),
        }
    }

    /// Trim generated text after stop sequences.
    fn trim_after_stops(text: &str, stops: &[String]) -> String {
        let mut shortest = text;
        for stop in stops {
            if let Some(idx) = shortest.find(stop.as_str()) {
                shortest = &shortest[..idx];
            }
        }
        shortest.trim().to_string()
    }
}

impl Backend for LlamaCliBackend {
    fn generate(
        &self,
        prompt: &str,
        params: &GenerationParams,
    ) -> Result<String> {
        let mut cmd = Command::new(&self.cli_path);
        cmd.arg("-m")
            .arg(&self.model_path)
            .arg("--ctx-size")
            .arg(self.ctx_size.to_string())
            .arg("--threads")
            .arg(self.n_threads.to_string())
            .arg("--temp")
            .arg(params.temperature.to_string())
            .arg("-n")
            .arg(params.max_tokens.to_string())
            .arg("-p")
            .arg(prompt)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let out = cmd
            .output()
            .map_err(|e| anyhow::anyhow!("Failed to run llama-cli: {}", e))?;

        let text = String::from_utf8_lossy(&out.stdout).to_string();
        let trimmed = Self::trim_after_stops(&text, &params.stop);
        // If llama-cli wrote nothing to stdout but logged to stderr, surface stderr.
        if trimmed.is_empty() && !out.stderr.is_empty() {
            let err = String::from_utf8_lossy(&out.stderr);
            if !err.contains("llama_new_context_with_model: n_ctx") {
                // Common non-fatal logging; ignore.
            }
        }
        Ok(trimmed)
    }
}
