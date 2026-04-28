static const char* model_lib_backend_tag(glm5_backend_t b) {
    switch (b) {
        case GLM5_BACKEND_CUDA: return "cuda";
        case GLM5_BACKEND_METAL: return "metal";
        case GLM5_BACKEND_VULKAN: return "vulkan";
        case GLM5_BACKEND_DIRECTML: return "directml";
        case GLM5_BACKEND_OPENCL: return "opencl";
        case GLM5_BACKEND_HIP: return "hip";
        case GLM5_BACKEND_LEVEL_ZERO: return "levelzero";
        case GLM5_BACKEND_SYCL: return "sycl";
        case GLM5_BACKEND_WEBGPU: return "webgpu";
        default: return "cpu";
    }
}

static const char* model_lib_ext() {
#ifdef _WIN32
    return ".dll";
#elif defined(__APPLE__)
    return ".dylib";
#else
    return ".so";
#endif
}

static bool path_exists_readable(const std::string& path) {
    return file_size_bytes(path, nullptr) != 0;
}

static bool path_is_directory(const std::string& path) {
    if (path.empty()) return false;
#ifdef _WIN32
    const std::wstring wide = storagellm::wide_path_from_utf8(path);
    if (wide.empty()) return false;
    WIN32_FILE_ATTRIBUTE_DATA data{};
    if (!GetFileAttributesExW(wide.c_str(), GetFileExInfoStandard, &data)) {
        return false;
    }
    return (data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
#else
    struct stat st{};
    if (stat(path.c_str(), &st) != 0) {
        return false;
    }
    return S_ISDIR(st.st_mode);
#endif
}

static int ensure_one_dir(const std::string& path) {
    if (path.empty()) return 0;
#ifdef _WIN32
    const std::wstring wide = storagellm::wide_path_from_utf8(path);
    if (wide.empty()) return 0;
    if (CreateDirectoryW(wide.c_str(), nullptr)) return 1;
    if (GetLastError() == ERROR_ALREADY_EXISTS && path_is_directory(path)) return 1;
#else
    if (mkdir(path.c_str(), 0755) == 0) return 1;
    if (errno == EEXIST && path_is_directory(path)) return 1;
#endif
    return 0;
}

static std::string model_lib_default_cache_dir(const glm5_pc_engine* e) {
    if (e && !e->model_lib_cache_dir.empty()) return e->model_lib_cache_dir;
    if (const char* env = std::getenv("GLM5_MODEL_LIB_CACHE")) return env;
    return ".glm5_model_lib_cache";
}

static std::string model_lib_filename(const glm5_pc_engine* e) {
    std::ostringstream os;
    os << "glm5_1_nvfp4_" << model_lib_backend_tag(e ? e->backend_caps.backend : GLM5_BACKEND_CPU)
       << model_lib_ext();
    return os.str();
}

static void model_lib_init_paths(glm5_pc_engine* e) {
    if (!e) return;
    if (e->config.model_lib_cache_dir) e->model_lib_cache_dir = e->config.model_lib_cache_dir;
    e->model_lib_cache_dir = model_lib_default_cache_dir(e);
    if (e->config.model_lib_compile_command) e->model_lib_compile_command = e->config.model_lib_compile_command;
    if (const char* env = std::getenv("GLM5_MODEL_LIB_COMPILE")) e->model_lib_compile_command = env;
    ensure_one_dir(e->model_lib_cache_dir);
    e->model_lib_path = e->config.model_lib_path ? e->config.model_lib_path :
        storagellm::path_join(e->model_lib_cache_dir, model_lib_filename(e));
    e->model_lib_source_path = e->model_lib_path + ".cpp";
}
