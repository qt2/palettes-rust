fn main() {
    #[cfg(feature = "gpu")]
    {
        cc::Build::new()
            .cuda(true)
            .flag("-gencode")
            .flag("arch=compute_89,code=sm_89")
            .files(["kernels/pedestrians.cu"])
            .compile("kernels");

        println!("cargo:rerun-if-changed=kernels");
    }
}
