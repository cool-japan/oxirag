fn main() {
    // napi-build wires up Node.js N-API linkage for cdylib (.node addon) builds.
    // Symbols are provided by the Node.js runtime when the addon is loaded.
    napi_build::setup();

    // For non-cdylib artifacts (test binaries, examples, oxirag-server) built
    // with the `nodejs` feature, the napi_* C symbols are absent from the build
    // environment.  Link a stub shared library that provides weak definitions so
    // the linker is satisfied.  The stubs are never called at runtime in
    // non-Node.js contexts.
    #[cfg(feature = "nodejs")]
    {
        use std::path::PathBuf;
        let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
        println!("cargo:rustc-link-search=native={}", manifest.display());
        println!("cargo:rustc-link-lib=dylib=napi_stub");
        println!("cargo:rerun-if-changed=napi_stub.c");
    }
}
