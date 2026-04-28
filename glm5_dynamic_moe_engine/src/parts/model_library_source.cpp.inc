static std::string model_lib_source_text(const glm5_pc_engine* e) {
    const glm5_storage_constants_t* c = glm5_storage_constants();
    std::ostringstream s;
    s << "#include <stdint.h>\n";
    s << "#if defined(_WIN32)\n#define GLM5_EXPORT __declspec(dllexport)\n";
    s << "#else\n#define GLM5_EXPORT __attribute__((visibility(\"default\")))\n#endif\n";
    s << "extern \"C\" {\n";
    s << "GLM5_EXPORT uint32_t glm5_model_lib_abi_version(){return 1u;}\n";
    s << "GLM5_EXPORT const char* glm5_model_lib_model_id(){return \"glm5.1-nvfp4-storage\";}\n";
    s << "GLM5_EXPORT const char* glm5_model_lib_backend(){return \""
      << model_lib_backend_tag(e ? e->backend_caps.backend : GLM5_BACKEND_CPU) << "\";}\n";
    s << "GLM5_EXPORT uint64_t glm5_model_lib_raw_tensor_bytes(){return "
      << (c ? c->raw_tensor_total_bytes : 0ull) << "ull;}\n";
    s << "GLM5_EXPORT uint32_t glm5_model_lib_layers(){return "
      << (c ? c->num_hidden_layers : 0u) << "u;}\n";
    s << "GLM5_EXPORT uint32_t glm5_model_lib_first_moe_layer(){return "
      << (c ? c->first_moe_layer : 0u) << "u;}\n";
    s << "GLM5_EXPORT uint32_t glm5_model_lib_last_moe_layer(){return "
      << (c ? c->last_moe_layer : 0u) << "u;}\n";
    s << "}\n";
    return s.str();
}

static int model_lib_write_source(glm5_pc_engine* e) {
    if (!e || e->model_lib_source_path.empty()) {
        return 0;
    }
    std::ofstream out;
    open_ofstream_path(out, e->model_lib_source_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return 0;
    }
    const std::string source = model_lib_source_text(e);
    out.write(source.data(), (std::streamsize)source.size());
    e->model_lib_generated = out.good() ? 1 : 0;
    return e->model_lib_generated;
}

static void replace_all(std::string& s, const std::string& from, const std::string& to) {
    if (from.empty()) return;
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
}

static int model_lib_try_compile(glm5_pc_engine* e) {
    if (!e || e->model_lib_compile_command.empty()) {
        return 0;
    }
    std::string cmd = e->model_lib_compile_command;
    replace_all(cmd, "{src}", e->model_lib_source_path);
    replace_all(cmd, "{out}", e->model_lib_path);
    replace_all(cmd, "{backend}", model_lib_backend_tag(e->backend_caps.backend));
    e->model_lib_compile_attempted = 1;
    int rc = -1;
#ifdef _WIN32
    STARTUPINFOW si = {};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi = {};
    const std::wstring wide_cmd = storagellm::wide_path_from_utf8(cmd);
    std::vector<wchar_t> cmdline(wide_cmd.begin(), wide_cmd.end());
    cmdline.push_back(L'\0');
    if (!wide_cmd.empty() &&
        CreateProcessW(nullptr, cmdline.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) {
        WaitForSingleObject(pi.hProcess, INFINITE);
        DWORD exitCode = 1;
        GetExitCodeProcess(pi.hProcess, &exitCode);
        rc = (int)exitCode;
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
    }
#else
    pid_t pid = fork();
    if (pid == 0) {
        // Fix 22: Enhanced POSIX arg parsing with single quotes + backslash escapes
        std::vector<std::string> args;
        std::string current_arg;
        bool in_double = false;
        bool in_single = false;
        bool escaped = false;
        for (char c : cmd) {
            if (escaped) {
                current_arg += c;
                escaped = false;
            } else if (c == '\\' && !in_single) {
                escaped = true;
            } else if (c == '\'' && !in_double) {
                in_single = !in_single;
            } else if (c == '"' && !in_single) {
                in_double = !in_double;
            } else if (c == ' ' && !in_double && !in_single) {
                if (!current_arg.empty()) {
                    args.push_back(current_arg);
                    current_arg.clear();
                }
            } else {
                current_arg += c;
            }
        }
        if (!current_arg.empty()) {
            args.push_back(current_arg);
        }
        
        std::vector<char*> c_args;
        for (auto& a : args) {
            c_args.push_back(&a[0]);
        }
        c_args.push_back(nullptr);
        
        execvp(c_args[0], c_args.data());
        _exit(127);
    } else if (pid > 0) {
        int status = 0;
        waitpid(pid, &status, 0);
        if (WIFEXITED(status)) {
            rc = WEXITSTATUS(status);
        }
    }
#endif
    e->model_lib_compile_succeeded = (rc == 0 && path_exists_readable(e->model_lib_path)) ? 1 : 0;
    return e->model_lib_compile_succeeded;
}
