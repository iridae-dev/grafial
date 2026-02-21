//! Build script for AOT compilation of flows.
//!
//! This build script can be used to compile Grafial flows to native code
//! at build time, producing optimized object files that can be linked
//! into the final binary.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    // Only run AOT compilation if the feature is enabled
    if env::var("CARGO_FEATURE_AOT").is_ok() {
        compile_flows();
    }
}

fn compile_flows() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    // Look for flow definitions in a specific directory
    let flows_dir = Path::new(&manifest_dir).join("flows");

    if !flows_dir.exists() {
        // No flows directory, skip compilation
        return;
    }

    println!("cargo:rerun-if-changed={}", flows_dir.display());

    // Create output directory for compiled flows
    let compiled_dir = Path::new(&out_dir).join("compiled_flows");
    fs::create_dir_all(&compiled_dir).unwrap();

    // Find all .gf (Grafial Flow) files
    let flow_files = find_flow_files(&flows_dir);

    if flow_files.is_empty() {
        return;
    }

    // Generate a manifest file listing all compiled flows
    let mut manifest = Vec::new();

    for flow_file in flow_files {
        let file_stem = flow_file.file_stem().unwrap().to_str().unwrap();
        let output_file = compiled_dir.join(format!("{}.o", file_stem));

        println!("cargo:rerun-if-changed={}", flow_file.display());

        // In a real implementation, we would:
        // 1. Parse the flow file
        // 2. Compile it using FlowCompiler
        // 3. Write the object file

        // For now, just record in manifest
        manifest.push(format!("{}:{}", file_stem, output_file.display()));
    }

    // Write manifest file
    let manifest_path = Path::new(&out_dir).join("flow_manifest.txt");
    fs::write(manifest_path, manifest.join("\n")).unwrap();

    // Set environment variable for runtime to find compiled flows
    println!(
        "cargo:rustc-env=GRAFIAL_COMPILED_FLOWS_DIR={}",
        compiled_dir.display()
    );
}

fn find_flow_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();

    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    if ext == "gf" || ext == "flow" {
                        files.push(path);
                    }
                }
            } else if path.is_dir() {
                // Recursively search subdirectories
                files.extend(find_flow_files(&path));
            }
        }
    }

    files
}
