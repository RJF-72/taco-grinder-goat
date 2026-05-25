// Context retriever + ranker.
// Memory-light: streaming TF-IDF or keyword overlap to pick top-K files.
// Plug-in point for heavier ML models later.

use lariat_capability::CapabilityProfile;
use std::collections::HashMap;
use std::path::Path;

/// Ranks project files against the current task to load into context.
pub trait ContextRanker: Send + Sync {
    fn rank(&self,
        task: &str,
        project_root: &Path,
        profile: &CapabilityProfile,
    ) -> Vec<RankedFile>;
}

#[derive(Debug, Clone)]
pub struct RankedFile {
    pub path: String,
    pub score: f64,
}

/// Streaming keyword TF-IDF ranker. No heavy embeddings, no GPU.
pub struct TfIdfRanker;

impl TfIdfRanker {
    pub fn new() -> Self { Self }

    fn tokenize(text: &str) -> Vec<String> {
        text
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty() && s.len() > 2)
            .map(|s| s.to_string())
            .collect()
    }
}

impl ContextRanker for TfIdfRanker {
    fn rank(&self,
        task: &str,
        project_root: &Path,
        profile: &CapabilityProfile,
    ) -> Vec<RankedFile> {
        let task_tokens = Self::tokenize(task);
        let max_files = match profile.tier {
            lariat_capability::Tier::Ultra => 12,
            lariat_capability::Tier::Standard => 8,
            lariat_capability::Tier::Lite => 5,
        };

        let mut scores: HashMap<String, f64> = HashMap::new();

        // Walk .rs files only, stream-read up to 8KB each.
        if let Ok(entries) = std::fs::read_dir(project_root) {
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_file() {
                    continue;
                }
                if path.extension().and_then(|s| s.to_str()) != Some("rs") {
                    continue;
                }
                let display = path.to_string_lossy().to_string();
                // Skip deps/target noise
                if display.contains("target\\") || display.contains("target/") {
                    continue;
                }
                let content = match std::fs::read(&path) {
                    Ok(bytes) => {
                        let limit = 8192usize.min(bytes.len());
                        String::from_utf8_lossy(&bytes[..limit]).to_string()
                    }
                    Err(_) => continue,
                };
                let file_tokens = Self::tokenize(&content);
                let overlap = task_tokens.iter().filter(|tok| file_tokens.contains(tok)).count() as f64;
                if overlap > 0.0 {
                    scores.insert(display, overlap);
                }
            }
        }

        let mut ranked: Vec<RankedFile> = scores
            .into_iter()
            .map(|(path, score)| RankedFile { path, score })
            .collect();
        ranked.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        ranked.truncate(max_files);
        ranked
    }
}
