// Template-based binary reading functions
// Eliminates duplication in GGUF read functions

static int moe_gguf_read_exact(std::ifstream& input, void* data, size_t bytes) {
    if (bytes == 0) {
        return 1;
    }
    input.read(static_cast<char*>(data), static_cast<std::streamsize>(bytes));
    return (size_t)input.gcount() == bytes;
}

// Template for reading fixed-size types
template<typename T>
static int moe_gguf_read(std::ifstream& input, T* out) {
    return out && moe_gguf_read_exact(input, out, sizeof(*out));
}

// Explicit instantiations for common types (for compatibility)
static int moe_gguf_read_u8(std::ifstream& input, uint8_t* out) {
    return moe_gguf_read(input, out);
}

static int moe_gguf_read_u16(std::ifstream& input, uint16_t* out) {
    return moe_gguf_read(input, out);
}

static int moe_gguf_read_u32(std::ifstream& input, uint32_t* out) {
    return moe_gguf_read(input, out);
}

static int moe_gguf_read_u64(std::ifstream& input, uint64_t* out) {
    return moe_gguf_read(input, out);
}

static int moe_gguf_skip_bytes(std::ifstream& input, uint64_t bytes) {
    while (bytes > 0) {
        const uint64_t chunk = std::min<uint64_t>(bytes, 0x3fffffffull);
        input.seekg((std::streamoff)chunk, std::ios::cur);
        if (!input.good()) {
            return 0;
        }
        bytes -= chunk;
    }
    return 1;
}
