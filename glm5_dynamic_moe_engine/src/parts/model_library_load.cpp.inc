typedef uint32_t (*glm5_model_lib_u32_fn)();

static void* model_lib_open_path(const std::string& path) {
    if (path.empty()) return nullptr;
#ifdef _WIN32
    return (void*)LoadLibraryA(path.c_str());
#else
    return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}

static void model_lib_unload(glm5_pc_engine* e) {
    if (!e || !e->model_lib_handle) return;
    glm5_close_shared_library(e->model_lib_handle);
    e->model_lib_handle = nullptr;
    e->model_lib_loaded = 0;
    e->model_lib_abi_version = 0;
}

static int model_lib_load(glm5_pc_engine* e) {
    if (!e || e->model_lib_path.empty() || !path_exists_readable(e->model_lib_path)) {
        return 0;
    }
    void* lib = model_lib_open_path(e->model_lib_path);
    if (!lib) {
        return 0;
    }
    auto abi = (glm5_model_lib_u32_fn)glm5_find_shared_symbol(lib, "glm5_model_lib_abi_version");
    const uint32_t version = abi ? abi() : 0;
    if (version != 1u) {
        glm5_close_shared_library(lib);
        return 0;
    }
    e->model_lib_handle = lib;
    e->model_lib_abi_version = version;
    e->model_lib_loaded = 1;
    return 1;
}

static int model_lib_prepare(glm5_pc_engine* e) {
    if (!e) return 0;
    model_lib_init_paths(e);
    if (model_lib_load(e)) return 1;
    if (!e->config.enable_model_lib_jit && e->model_lib_compile_command.empty()) return 0;
    if (!model_lib_write_source(e)) return 0;
    if (model_lib_try_compile(e)) return model_lib_load(e);
    return 0;
}
