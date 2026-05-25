// Physical Model I/O
// Stream data to/from disk. Keep RAM clean.

use std::fs;
use std::path::{Path, PathBuf};
use crate::{Step, TrajectorySummary};

/// Ensure directory exists.
pub fn ensure_dir(p: &Path) -> std::io::Result<()> {
    if !p.exists() {
        fs::create_dir_all(p)?;
    }
    Ok(())
}

/// Write a step JSON to disk.
pub fn write_step(dir: &Path, name: &str, step: &Step) -> std::io::Result<()> {
    let path = dir.join(name);
    let json = serde_json::to_string_pretty(step).map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, e)
    })?;
    fs::write(path, json)?;
    Ok(())
}

/// Write a trajectory summary to disk.
pub fn write_summary(dir: &Path, summary: &TrajectorySummary) -> std::io::Result<()> {
    let path = dir.join("summary.json");
    let json = serde_json::to_string_pretty(summary).map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, e)
    })?;
    fs::write(path, json)?;
    Ok(())
}

/// List trajectory directories under a root.
pub fn list_trajectory_dirs(root: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            out.push(path);
        }
    }
    Ok(out)
}