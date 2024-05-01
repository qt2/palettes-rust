fn main() {
    #[cfg(feature = "gpu")]
    {
        cc::Build::new()
            .cuda(true)
            .flag("-gencode")
            .flag("arch=compute_20,code=sm_20")
            .files(["kernels/pedestrians.cu"])
            .compile("kernels");

        println!("cargo:rerun-if-changed=kernels");
    }
}
