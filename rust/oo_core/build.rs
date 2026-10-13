//! The ABI of the interpreter this artefact is built for, stamped into it: the loader reads the stamp in the file
//! before it loads anything, and loads only an artefact built for the running interpreter's ABI (one built for
//! another may crash the process at its first call, as a 3.13 artefact does under 3.14).
//!
//! The interpreter is the one pyo3 0.22 builds against, found in its own order: PYO3_PYTHON (which
//! scripts/build_oo_core.sh sets); else the active virtual environment or the active conda environment, when exactly
//! one of them is active; else `python`, then `python3`. The stamp is its `sysconfig.get_config_var("SOABI")` (for
//! example `cpython-313-x86_64-linux-gnu`: the version, the ABI flags such as the free-threaded `t`, the platform).
//! It is `unknown`, which no interpreter matches, when the interpreter cannot be asked or when pyo3 is configured
//! by a file (PYO3_CONFIG_FILE) this script does not read.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn ask(python: &PathBuf) -> Option<String> {
    let out = Command::new(python)
        .args(["-I", "-c", "import sysconfig; print(sysconfig.get_config_var('SOABI') or '')"])
        .output()
        .ok()?;
    let tag = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let plain = !tag.is_empty() && tag.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    (out.status.success() && plain).then_some(tag)
}

fn stamp() -> Option<String> {
    if env::var_os("PYO3_CONFIG_FILE").is_some() {
        return None;
    }
    if let Some(python) = env::var_os("PYO3_PYTHON") {
        return ask(&PathBuf::from(python));
    }
    let environments: Vec<PathBuf> = ["VIRTUAL_ENV", "CONDA_PREFIX"]
        .iter()
        .filter_map(|name| env::var_os(name))
        .map(|root| PathBuf::from(root).join("bin").join("python"))
        .collect();
    if environments.len() == 1 {
        return ask(&environments[0]);
    }
    ["python", "python3"].iter().find_map(|name| ask(&PathBuf::from(name)))
}

fn main() {
    for name in ["PYO3_PYTHON", "PYO3_CONFIG_FILE", "VIRTUAL_ENV", "CONDA_PREFIX", "PATH"] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    let tag = stamp().unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=OO_CORE_PYTHON_TAG={tag}");
}
