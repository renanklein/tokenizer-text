fn main() {
    println!("cargo:rustc-link-search=native=<path_to_libtorch>");
    println!("cargo:rustc-link-lib=torch_cuda");
    println!("cargo:rustc-link-arg=-Wl,--no-as-needed,-ltorch_cuda");
}
