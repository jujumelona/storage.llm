static int compiled_nvfp4_device_kernels_available() {
#ifdef moe_ENABLE_NVFP4_DEVICE_KERNELS
    return 1;
#else
    return 0;
#endif
}

static int compiled_scale4_device_fusion_available() {
#ifdef moe_ENABLE_SCALE4_DEVICE_FUSION
    return 1;
#else
    return 0;
#endif
}
