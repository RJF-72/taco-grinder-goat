// LARIAT CAPABILITY PROFILE
// Defines per-machine hardware fingerprint + tiered limits.
// All decisions (model, quantization, context, features) derive from this profile.

use serde::{Deserialize, Serialize};

mod probe;

pub use probe::probe_system;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Tier {
    Ultra,
    Standard,
    Lite,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLimits {
    pub max_model_mem_mb: u32,
    pub max_context_tokens: u32,
    pub max_kv_cache_mb: u32,
    pub max_parallel_tasks: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureFlags {
    pub vision_enabled: bool,
    pub high_precision_allowed: bool,
}

/// Capability profile derived at startup + updated periodically.
/// Drives every downstream decision to stay off memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityProfile {
    // Raw hardware readings
    pub ram_total_mb: u32,
    pub ram_free_mb: u32,
    pub vram_total_mb: u32,
    pub vram_free_mb: u32,
    pub cpu_cores: u32,
    pub cpu_freq_ghz: f32,
    pub disk_type: DiskType,
    pub disk_free_gb: u32,
    pub disk_read_mbps: u32,
    pub disk_write_mbps: u32,
    pub network_bandwidth_mbps: u32,
    pub power_mode: PowerMode,

    // Derived
    pub tier: Tier,
    pub limits: MemoryLimits,
    pub features: FeatureFlags,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DiskType {
    Ssd,
    Hdd,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PowerMode {
    Plugged,
    Battery,
}

impl CapabilityProfile {
    /// Probe the current machine and compute the profile.
    /// This is the canonical entry point Lariat calls on startup.
    pub fn probe() -> Self {
        let sys = probe::probe_system();
        let (vram_total, vram_free) = probe::probe_gpu_wmi().unwrap_or((0, 0));
        let ram_total = sys.ram_total_mb;
        let ram_free = sys.ram_free_mb;
        let cpu_cores = sys.cpu_cores;
        let tier = Self::derive_tier(ram_total, ram_free, vram_total, vram_free);
        let limits = Self::derive_limits(&tier, ram_free, vram_free);
        let features = Self::derive_features(&tier, &limits);
        Self {
            ram_total_mb: ram_total,
            ram_free_mb: ram_free,
            vram_total_mb: vram_total,
            vram_free_mb: vram_free,
            cpu_cores,
            cpu_freq_ghz: 3.2,
            disk_type: sys.disk_type,
            disk_free_gb: sys.disk_free_gb,
            disk_read_mbps: sys.disk_read_mbps,
            disk_write_mbps: sys.disk_write_mbps,
            network_bandwidth_mbps: 100,
            power_mode: sys.power_mode,
            tier,
            limits,
            features,
        }
    }

    fn derive_tier(ram_total: u32, _ram_free: u32, vram_total: u32, _vram_free: u32) -> Tier {
        if vram_total >= 12000 || ram_total >= 64000 {
            Tier::Ultra
        } else if vram_total >= 4096 || ram_total >= 16000 {
            Tier::Standard
        } else {
            Tier::Lite
        }
    }

    /// Limits are memory-first.
    fn derive_limits(tier: &Tier, ram_free: u32, vram_free: u32) -> MemoryLimits {
        match tier {
            Tier::Ultra => MemoryLimits {
                max_model_mem_mb: 5_000,
                max_context_tokens: 16_000,
                max_kv_cache_mb: 3_000,
                max_parallel_tasks: 4,
            },
            Tier::Standard => MemoryLimits {
                max_model_mem_mb: 3_000,
                max_context_tokens: 8_000,
                max_kv_cache_mb: 1_500,
                max_parallel_tasks: 2,
            },
            Tier::Lite => MemoryLimits {
                max_model_mem_mb: 1_500,
                max_context_tokens: 4_096,
                max_kv_cache_mb: 512,
                max_parallel_tasks: 1,
            },
        }
    }

    fn derive_features(tier: &Tier, limits: &MemoryLimits) -> FeatureFlags {
        match tier {
            Tier::Ultra => FeatureFlags {
                vision_enabled: true,
                high_precision_allowed: true,
            },
            Tier::Standard => FeatureFlags {
                vision_enabled: true,
                high_precision_allowed: false,
            },
            Tier::Lite => FeatureFlags {
                vision_enabled: false,
                high_precision_allowed: false,
            },
        }
    }

    /// Dynamic runtime adjustment: if memory pressure is detected, downgrade in-session.
    pub fn tighten_for_pressure(&mut self) {
        if self.tier != Tier::Lite {
            self.tier = match self.tier {
                Tier::Ultra => Tier::Standard,
                Tier::Standard => Tier::Lite,
                Tier::Lite => Tier::Lite,
            };
            self.limits = Self::derive_limits(&self.tier, self.ram_free_mb, self.vram_free_mb);
            self.features = Self::derive_features(&self.tier, &self.limits);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_derives_correctly() {
        let p = CapabilityProfile::probe();
        match p.tier {
            Tier::Standard => {
                assert!(p.limits.max_model_mem_mb <= 3_000);
            }
            _ => {}
        }
    }
}
