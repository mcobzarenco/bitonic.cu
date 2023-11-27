use regex::Regex;
use std::{env, path::PathBuf, process::Command};

fn main() {
    // Tell cargo to invalidate the built crate whenever files of interest changes.
    println!("cargo:rerun-if-changed={}", "cuda");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // SM 8.6 (Ampere architecture)
    let arch = "compute_86";
    let code = "sm_86";

    // build the cuda kernels
    let cuda_src = PathBuf::from("src/cuda/bitonic_full.cu");
    let ptx_file = out_dir.join("bitonic.ptx");

    let nvcc_status = Command::new("nvcc")
        .arg("-O3")
        .arg("-Xptxas")
        .arg("--use_fast_math")
        .arg("-ptx")
        .arg("-dopt=on")
        .arg("-o")
        .arg(&ptx_file)
        .arg(&cuda_src)
        .arg(format!("-arch={}", arch))
        .arg(format!("-code={}", code))
        .status()
        .unwrap();

    assert!(
        nvcc_status.success(),
        "Failed to compile CUDA source to PTX."
    );

    let bindings = bindgen::Builder::default()
        .header("src/cuda/struct.h")
        .derive_copy(true)
        .derive_partialeq(true)
        .generate()
        .expect("Unable to generate bindings");

    // we need to make modifications to the generated code
    let generated_bindings = bindings.to_string();

    // Regex to find raw pointers to float and replace them with CudaSlice<f32>
    // You can copy this regex to add/modify other types of pointers, for example "*mut i32"
    let pointer_regex = Regex::new(r"\*mut f32").unwrap();
    let modified_bindings = pointer_regex.replace_all(&generated_bindings, "CudaSlice<f32>");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    std::fs::write(out_path.join("bindings.rs"), modified_bindings.as_bytes())
        .expect("Failed to write bindings");
}
