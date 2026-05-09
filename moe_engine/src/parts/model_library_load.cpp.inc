typedef uint32_t (*moe_model_lib_u32_fn)();
typedef const void* (*moe_model_lib_const_ptr_fn)();
typedef const char* (*moe_model_lib_cstr_fn)();

static void* model_lib_open_path(const std::string& path) {
    if (path.empty()) return nullptr;
#ifdef _WIN32
    return (void*)LoadLibraryA(path.c_str());
#else
    return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}

static void model_lib_unload(moe_pc_engine* e) {
    if (!e || !e->model_lib_handle) return;
    moe_close_shared_library(e->model_lib_handle);
    e->model_lib_handle = nullptr;
    e->model_lib_expert_batch_f32 = nullptr;
    e->model_lib_expert_batch_contiguous_f32 = nullptr;
    e->model_lib_expert_batch_device_f32 = nullptr;
    e->model_lib_grouped_expert_triplets_f32 = nullptr;
    e->model_lib_grouped_expert_triplets_device_f32 = nullptr;
    e->model_lib_grouped_expert_triplets_indexed_f32 = nullptr;
    e->model_lib_grouped_expert_triplets_indexed_device_f32 = nullptr;
    e->model_lib_prefill_activation_acquire_f32 = nullptr;
    e->model_lib_prefill_activation_release_f32 = nullptr;
    e->model_lib_gpu_expert_scratch_wait_ns = nullptr;
    e->model_lib_gpu_expert_scratch_backpressure_count = nullptr;
    e->model_lib_fast_backend_mask = nullptr;
    e->model_lib_l0_mmap_checksum_spv = nullptr;
    e->model_lib_l0_mmap_checksum_spv_size = 0;
    e->model_lib_l0_mmap_checksum_kernel_name = nullptr;
    e->model_lib_loaded = 0;
    e->model_lib_abi_version = 0;
}

static int model_lib_load(moe_pc_engine* e) {
    if (!e || e->model_lib_path.empty() || !path_exists_readable(e->model_lib_path)) {
        return 0;
    }
    void* lib = model_lib_open_path(e->model_lib_path);
    if (!lib) {
        return 0;
    }
    auto abi = (moe_model_lib_u32_fn)moe_find_shared_symbol(lib, "moe_model_lib_abi_version");
    const uint32_t version = abi ? abi() : 0;
    if (version != 1u) {
        moe_close_shared_library(lib);
        return 0;
    }
    e->model_lib_handle = lib;
    e->model_lib_abi_version = version;
    void* expert_batch =
        moe_find_shared_symbol(lib, "moe_model_lib_run_expert_triplet_batch_f32");
    void* expert_batch_contiguous =
        moe_find_shared_symbol(lib, "moe_model_lib_run_expert_triplet_batch_contiguous_f32");
    void* expert_batch_device =
        moe_find_shared_symbol(lib, "moe_model_lib_run_expert_triplet_batch_device_f32");
    void* grouped_expert_triplets =
        moe_find_shared_symbol(lib, "moe_model_lib_run_grouped_expert_triplets_f32");
    void* grouped_expert_triplets_device =
        moe_find_shared_symbol(lib, "moe_model_lib_run_grouped_expert_triplets_device_f32");
    void* grouped_expert_triplets_indexed =
        moe_find_shared_symbol(lib, "moe_model_lib_run_grouped_expert_triplets_indexed_f32");
    void* grouped_expert_triplets_indexed_device =
        moe_find_shared_symbol(lib, "moe_model_lib_run_grouped_expert_triplets_indexed_device_f32");
    void* prefill_activation_acquire =
        moe_find_shared_symbol(lib, "moe_model_lib_prefill_activation_acquire_f32");
    void* prefill_activation_release =
        moe_find_shared_symbol(lib, "moe_model_lib_prefill_activation_release_f32");
    void* scratch_wait_ns =
        moe_find_shared_symbol(lib, "moe_model_lib_gpu_expert_scratch_wait_ns");
    void* scratch_backpressure =
        moe_find_shared_symbol(lib, "moe_model_lib_gpu_expert_scratch_backpressure_count");
    void* fast_backend_mask =
        moe_find_shared_symbol(lib, "moe_model_lib_fast_backend_mask");
    auto expert_batch_available =
        (moe_model_lib_u32_fn)moe_find_shared_symbol(lib, "moe_model_lib_has_expert_triplet_batch_f32");
    if (expert_batch_available && expert_batch_available() == 0u) {
        expert_batch = nullptr;
        expert_batch_contiguous = nullptr;
        expert_batch_device = nullptr;
        grouped_expert_triplets = nullptr;
        grouped_expert_triplets_device = nullptr;
        grouped_expert_triplets_indexed = nullptr;
        grouped_expert_triplets_indexed_device = nullptr;
        prefill_activation_acquire = nullptr;
        prefill_activation_release = nullptr;
    }
    const uint32_t fast_mask =
        fast_backend_mask ? ((moe_model_lib_u32_fn)fast_backend_mask)() : 0u;
    if (e->io_config.internal_verified_backend_only &&
        !moe_fast_backend_mask_supports_backend(e->backend_caps.backend, fast_mask)) {
        expert_batch = nullptr;
        expert_batch_contiguous = nullptr;
        expert_batch_device = nullptr;
        grouped_expert_triplets = nullptr;
        grouped_expert_triplets_device = nullptr;
        grouped_expert_triplets_indexed = nullptr;
        grouped_expert_triplets_indexed_device = nullptr;
        prefill_activation_acquire = nullptr;
        prefill_activation_release = nullptr;
    }
    e->model_lib_expert_batch_f32 = expert_batch;
    e->model_lib_expert_batch_contiguous_f32 = expert_batch_contiguous;
    e->model_lib_expert_batch_device_f32 = expert_batch_device;
    e->model_lib_grouped_expert_triplets_f32 = grouped_expert_triplets;
    e->model_lib_grouped_expert_triplets_device_f32 = grouped_expert_triplets_device;
    e->model_lib_grouped_expert_triplets_indexed_f32 = grouped_expert_triplets_indexed;
    e->model_lib_grouped_expert_triplets_indexed_device_f32 = grouped_expert_triplets_indexed_device;
    e->model_lib_prefill_activation_acquire_f32 = prefill_activation_acquire;
    e->model_lib_prefill_activation_release_f32 = prefill_activation_release;
    e->model_lib_gpu_expert_scratch_wait_ns = scratch_wait_ns;
    e->model_lib_gpu_expert_scratch_backpressure_count = scratch_backpressure;
    e->model_lib_fast_backend_mask = fast_backend_mask;
    auto l0_spv_size =
        (moe_model_lib_u32_fn)moe_find_shared_symbol(lib, "moe_model_lib_l0_mmap_checksum_spv_size");
    auto l0_spv =
        (moe_model_lib_const_ptr_fn)moe_find_shared_symbol(lib, "moe_model_lib_l0_mmap_checksum_spv");
    auto l0_kernel_name =
        (moe_model_lib_cstr_fn)moe_find_shared_symbol(lib, "moe_model_lib_l0_mmap_checksum_kernel_name");
    e->model_lib_l0_mmap_checksum_spv_size = l0_spv_size ? l0_spv_size() : 0u;
    e->model_lib_l0_mmap_checksum_spv =
        (l0_spv && e->model_lib_l0_mmap_checksum_spv_size > 0) ? l0_spv() : nullptr;
    e->model_lib_l0_mmap_checksum_kernel_name = l0_kernel_name ? l0_kernel_name() : nullptr;
    e->model_lib_loaded = 1;
    return 1;
}

static int model_lib_prepare(moe_pc_engine* e) {
    if (!e) return 0;
    model_lib_init_paths(e);
    if (model_lib_load(e)) return 1;
    if (!e->config.enable_model_lib_jit && e->model_lib_compile_command.empty()) return 0;
    if (!model_lib_write_source(e)) return 0;
    if (model_lib_try_compile(e)) return model_lib_load(e);
    return 0;
}
