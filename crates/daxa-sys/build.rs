use std::env;
use std::path::{Path, PathBuf};

fn main() {
    dotenvy::dotenv();
    let static_crt = env::var("CARGO_ENCODED_RUSTFLAGS")
        .unwrap_or_default()
        .contains("target-feature=+crt-static");

    let static_crt = &format!("-DDAXA_USE_STATIC_CRT={}", static_crt as usize);

    let mut daxa_dir = workspace_dir();
    daxa_dir.push("lib/daxa");

    let mut module_dir = daxa_dir.clone();
    module_dir.push("cmake/modules");

    let mut daxa_include = daxa_dir.clone();
    daxa_include.push("include");
    let daxa_include_flag = format!("-I{}", daxa_include.display());

    let module_path = &format!("-DCMAKE_MODULE_PATH={module_dir:?}");

    println!("cargo:warning={daxa_include_flag}");

    let dst = cmake::Config::new(daxa_dir)
        .build_target("daxa")
        .profile(get_profile())
        .configure_arg("-DBUILD_SHARED_LIBS=OFF")
        .configure_arg("-DDAXA_USE_VCPKG=ON")
        .configure_arg(module_path)
        .configure_arg(static_crt)
        .build();

    let vulkan_path =
        env::var("VULKAN_SDK").expect("VULKAN_SDK must be specified for daxa to build");

    println!("cargo:rustc-link-search=native={vulkan_path}/lib",);
    println!("cargo:rustc-link-search=native={}/build/", dst.display(),);
    println!(
        "cargo:rustc-link-search=native={}/build/{}",
        dst.display(),
        get_profile()
    );
    println!(
        "cargo:rustc-link-search=native={}/build/vcpkg_installed/{}/lib",
        dst.display(),
        get_vcpkg_os_dir()
    );
    println!("cargo:rustc-link-search=native={}/lib", vulkan_path);
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=dylib=X11");
    println!("cargo:rustc-link-lib=static=daxa");
    println!("cargo:rustc-link-lib=static=fmt");
    #[cfg(target_os = "windows")]
    println!("cargo:rustc-link-lib=static=vulkan-1");
    #[cfg(not(target_os = "windows"))]
    println!("cargo:rustc-link-lib=dylib=vulkan");
    println!("cargo:rerun-if-changed=src/daxa.h");
    println!("cargo:rerun-if-changed=daxa");
    println!("cargo:rustc-link-arg=-static");

    let vcpkg_includes = format!(
        "-I{}/build/vcpkg_installed/{}/include",
        dst.display(),
        get_vcpkg_os_dir()
    );
    let bindings = bindgen::Builder::default()
        .clang_arg("--target=x86_64-unknown-linux-gnu")
        .clang_arg("--language=c")
        .clang_arg("-std=c23")
        .clang_arg(daxa_include_flag)
        .clang_arg(vcpkg_includes)
        .header("src/daxa.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

#[cfg(target_os = "windows")]
fn get_vcpkg_os_dir() -> &'static str {
    "x64-windows"
}
#[cfg(target_os = "linux")]
fn get_vcpkg_os_dir() -> &'static str {
    "x64-linux"
}

#[cfg(debug_assertions)]
fn get_profile() -> &'static str {
    "debug"
}

#[cfg(not(debug_assertions))]
fn get_profile() -> &'static str {
    "release"
}

fn workspace_dir() -> PathBuf {
    let output = std::process::Command::new(env!("CARGO"))
        .arg("locate-project")
        .arg("--workspace")
        .arg("--message-format=plain")
        .output()
        .unwrap()
        .stdout;
    let cargo_path = Path::new(std::str::from_utf8(&output).unwrap().trim());
    cargo_path.parent().unwrap().to_path_buf()
}
