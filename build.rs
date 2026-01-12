fn main() {
    println!("cargo:rustc-link-search=native=<lib_path>");
    println!("cargo:rustc-link-lib=torch_cuda");
    println!("cargo:rustc-link-arg=-Wl,--no-as-needed,-ltorch_cuda");
}
