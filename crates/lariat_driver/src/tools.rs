// Tool executor with JSON schema validation.
// The model emits tool calls; this validates them and runs them.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub required_args: Vec<String>,
    pub optional_args: Vec<String>,
    pub returns: String,
}

/// Executor validates JSON tool calls against known schema, then dispatches.
pub struct ToolExecutor {
    registry: HashMap<String, ToolSpec>,
}

impl ToolExecutor {
    pub fn with_defaults() -> Self {
        let mut registry = HashMap::new();
        registry.insert(
            "open_file".into(),
            ToolSpec {
                name: "open_file".into(),
                required_args: vec!["path".into()],
                optional_args: vec!["offset".into(), "limit".into()],
                returns: "file_content".into(),
            },
        );
        registry.insert(
            "search".into(),
            ToolSpec {
                name: "search".into(),
                required_args: vec!["query".into()],
                optional_args: vec!["path_glob".into(), "max_results".into()],
                returns: "results".into(),
            },
        );
        registry.insert(
            "edit_file".into(),
            ToolSpec {
                name: "edit_file".into(),
                required_args: vec!["path".into(), "old_text".into(), "new_text".into()],
                optional_args: vec![],
                returns: "diff".into(),
            },
        );
        registry.insert(
            "cargo_check".into(),
            ToolSpec {
                name: "cargo_check".into(),
                required_args: vec![],
                optional_args: vec!["target".into()],
                returns: "compile_output".into(),
            },
        );
        registry.insert(
            "run_tests".into(),
            ToolSpec {
                name: "run_tests".into(),
                required_args: vec![],
                optional_args: vec!["filter".into()],
                returns: "test_output".into(),
            },
        );
        registry.insert(
            "summarize".into(),
            ToolSpec {
                name: "summarize".into(),
                required_args: vec!["target".into()],
                optional_args: vec!["max_words".into()],
                returns: "summary".into(),
            },
        );
        registry.insert(
            "ask_user".into(),
            ToolSpec {
                name: "ask_user".into(),
                required_args: vec!["question".into()],
                optional_args: vec![],
                returns: "answer".into(),
            },
        );
        Self { registry }
    }

    /// Validate a tool call JSON against registry schema.
    pub fn validate(&self,
        tool_name: &str,
        args: &HashMap<String, serde_json::Value>,
    ) -> Result<(), ToolError> {
        let spec = self.registry.get(tool_name).ok_or_else(|| ToolError::UnknownTool(tool_name.into()))?;
        for req in &spec.required_args {
            if !args.contains_key(req) {
                return Err(ToolError::MissingArg {
                    tool: tool_name.into(),
                    arg: req.into(),
                });
            }
        }
        Ok(())
    }

    pub fn list_tools(&self) -> Vec<&ToolSpec> {
        self.registry.values().collect()
    }
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum ToolError {
    #[error("Unknown tool: {0}")]
    UnknownTool(String),
    #[error("Missing required argument '{arg}' for tool '{tool}'")]
    MissingArg { tool: String, arg: String },
    #[error("Execution failed: {0}")]
    ExecutionFailed(String),
}

/// Dispatch a tool call to an actual operation.
/// This is a stub; real implementations would shell out or call into the OS.
pub fn dispatch(tool: &str, args: &HashMap<String, serde_json::Value>) -> Result<String, ToolError> {
    match tool {
        "open_file" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let contents = std::fs::read_to_string(path)
                .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
            Ok(contents)
        }
        "cargo_check" => {
            Ok("cargo check: ok".into())
        }
        "run_tests" => {
            Ok("tests: all passed".into())
        }
        "summarize" => {
            let target = args.get("target").and_then(|v| v.as_str()).unwrap_or("");
            Ok(format!("Summary of {}", target))
        }
        _ => Err(ToolError::UnknownTool(tool.into())),
    }
}
