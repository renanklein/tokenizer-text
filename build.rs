fn main() {
    println!("cargo:rustc-link-search=native=/home/renanklein/libtorch_cu128");
    println!("cargo:rustc-link-lib=torch_cuda");
    println!("cargo:rustc-link-arg=-Wl,--no-as-needed,-ltorch_cuda");
}
