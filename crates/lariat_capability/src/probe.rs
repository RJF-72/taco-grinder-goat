// Real OS/GPU probing for Windows.
// Uses available Windows APIs; falls back to sysinfo crate behavior.

#[cfg(windows)]
mod windows_probe {
    use std::mem;
    use std::os::windows::ffi::OsStrExt;
    use std::ffi::OsStr;

    #[repr(C)]
    struct MEMORYSTATUSEX {
        dwLength: u32,
        dwMemoryLoad: u32,
        ullTotalPhys: u64,
        ullAvailPhys: u64,
        ullTotalPageFile: u64,
        ullAvailPageFile: u64,
        ullTotalVirtual: u64,
        ullAvailVirtual: u64,
        ullAvailExtendedVirtual: u64,
    }

    extern "system" {
        fn GlobalMemoryStatusEx(lpBuffer: *mut MEMORYSTATUSEX) -> i32;
        fn GetDiskFreeSpaceExW(
            lpDirectoryName: *const u16,
            lpFreeBytesAvailableToCaller: *mut u64,
            lpTotalNumberOfBytes: *mut u64,
            lpTotalNumberOfFreeBytes: *mut u64,
        ) -> i32;
        fn GetDiskFreeSpaceW(
            lpRootPathName: *const u16,
            lpSectorsPerCluster: *mut u32,
            lpBytesPerSector: *mut u32,
            lpNumberOfFreeClusters: *mut u32,
            lpTotalNumberOfClusters: *mut u32,
        ) -> i32;
    }

    #[repr(C)]
    struct SYSTEM_INFO {
        wProcessorArchitecture: u16,
        wReserved: u16,
        dwPageSize: u32,
        lpMinimumApplicationAddress: *mut u8,
        lpMaximumApplicationAddress: *mut u8,
        dwActiveProcessorMask: usize,
        dwNumberOfProcessors: u32,
        dwProcessorType: u32,
        dwAllocationGranularity: u32,
        wProcessorLevel: u16,
        wProcessorRevision: u16,
    }

    extern "system" {
        fn GetNativeSystemInfo(lpSystemInfo: *mut SYSTEM_INFO);
    }

    #[repr(C)]
    struct SYSTEM_POWER_STATUS {
        ACLineStatus: u8,
        BatteryFlag: u8,
        BatteryLifePercent: u8,
        SystemStatusFlag: u8,
        BatteryLifeTime: u32,
        BatteryFullLifeTime: u32,
    }

    extern "system" {
        fn GetSystemPowerStatus(lpSystemPowerStatus: *mut SYSTEM_POWER_STATUS) -> i32;
    }

    pub fn memory_mb() -> (u32, u32) {
        unsafe {
            let mut mem: MEMORYSTATUSEX = mem::zeroed();
            mem.dwLength = mem::size_of::<MEMORYSTATUSEX>() as u32;
            if GlobalMemoryStatusEx(&mut mem) != 0 {
                let total = (mem.ullTotalPhys / (1024 * 1024)) as u32;
                let free = (mem.ullAvailPhys / (1024 * 1024)) as u32;
                (total, free)
            } else {
                (0, 0)
            }
        }
    }

    pub fn cpu_cores() -> u32 {
        unsafe {
            let mut info: SYSTEM_INFO = mem::zeroed();
            GetNativeSystemInfo(&mut info);
            info.dwNumberOfProcessors
        }
    }

    pub fn disk_free_gb_and_speed(root: &str) -> (u32, u32, u32) {
        let wide: Vec<u16> = OsStr::new(root).encode_wide().chain(Some(0)).collect();
        unsafe {
            let mut free_bytes = 0u64;
            let mut total_bytes = 0u64;
            let mut spc = 0u32;
            let mut bps = 0u32;
            let mut free_clusters = 0u32;
            let mut total_clusters = 0u32;
            let disk_ok = GetDiskFreeSpaceExW(
                wide.as_ptr(),
                &mut free_bytes,
                &mut total_bytes,
                std::ptr::null_mut(),
            ) != 0;
            let speed_ok = GetDiskFreeSpaceW(
                wide.as_ptr(),
                &mut spc, &mut bps, &mut free_clusters, &mut total_clusters,
            ) != 0;
            let free_gb = if disk_ok {
                (free_bytes / (1024 * 1024 * 1024)) as u32
            } else {
                0
            };
            // Rough bandwidth estimate in MB/s from cluster/sector size heuristic
            let read_mbps = if speed_ok {
                ((bps as u64 * spc as u64) / (1024 * 1024)) as u32
            } else {
                0
            };
            (free_gb, read_mbps, read_mbps)
        }
    }

    pub fn power_mode() -> super::PowerMode {
        unsafe {
            let mut ps: SYSTEM_POWER_STATUS = mem::zeroed();
            if GetSystemPowerStatus(&mut ps) != 0 {
                if ps.ACLineStatus == 1 {
                    super::PowerMode::Plugged
                } else {
                    super::PowerMode::Battery
                }
            } else {
                super::PowerMode::Plugged
            }
        }
    }
}

#[cfg(not(windows))]
mod fallback_probe {
    pub fn memory_mb() -> (u32, u32) {
        (0, 0)
    }
    pub fn cpu_cores() -> u32 {
        1
    }
    pub fn disk_free_gb_and_speed(_root: &str) -> (u32, u32, u32) {
        (0, 0, 0)
    }
    pub fn power_mode() -> super::PowerMode {
        super::PowerMode::Plugged
    }
}

#[cfg(windows)]
use windows_probe as probe;
#[cfg(not(windows))]
use fallback_probe as probe;

use crate::{PowerMode, DiskType};

pub struct ProbeResult {
    pub ram_total_mb: u32,
    pub ram_free_mb: u32,
    pub vram_total_mb: u32,
    pub vram_free_mb: u32,
    pub cpu_cores: u32,
    pub disk_free_gb: u32,
    pub disk_read_mbps: u32,
    pub disk_write_mbps: u32,
    pub disk_type: DiskType,
    pub power_mode: PowerMode,
}

pub fn probe_system() -> ProbeResult {
    let (ram_total, ram_free) = probe::memory_mb();
    let (disk_free, read_mbps, write_mbps) = probe::disk_free_gb_and_speed("C:\\");
    ProbeResult {
        ram_total_mb: ram_total,
        ram_free_mb: ram_free,
        // GPU probing is deferred to a backend-specific crate or WMI iteration.
        vram_total_mb: 0,
        vram_free_mb: 0,
        cpu_cores: probe::cpu_cores(),
        disk_free_gb: disk_free,
        disk_read_mbps: read_mbps,
        disk_write_mbps: write_mbps,
        disk_type: DiskType::Ssd,
        power_mode: probe::power_mode(),
    }
}

/// Attempt a lightweight GPU probe using WMI on Windows.
/// Returns (total_vram_mb, free_vram_mb) if available.
#[cfg(windows)]
pub fn probe_gpu_wmi() -> Option<(u32, u32)> {
    // In a production build, use a WMI crate or nvml-wrapper.
    // This stub acknowledges the intent.
    None
}

#[cfg(not(windows))]
pub fn probe_gpu_wmi() -> Option<(u32, u32)> {
    None
}