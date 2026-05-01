// GGUF type definitions and I/O functions
// Handles GGUF type system, reading/skipping values, and type conversions

enum {
    moe_GGUF_TYPE_UINT8 = 0,
    moe_GGUF_TYPE_INT8 = 1,
    moe_GGUF_TYPE_UINT16 = 2,
    moe_GGUF_TYPE_INT16 = 3,
    moe_GGUF_TYPE_UINT32 = 4,
    moe_GGUF_TYPE_INT32 = 5,
    moe_GGUF_TYPE_FLOAT32 = 6,
    moe_GGUF_TYPE_BOOL = 7,
    moe_GGUF_TYPE_STRING = 8,
    moe_GGUF_TYPE_ARRAY = 9,
    moe_GGUF_TYPE_UINT64 = 10,
    moe_GGUF_TYPE_INT64 = 11,
    moe_GGUF_TYPE_FLOAT64 = 12
};

static int moe_gguf_fixed_type_size(uint32_t type, uint64_t* out) {
    if (!out) {
        return 0;
    }
    switch (type) {
        case moe_GGUF_TYPE_UINT8:
        case moe_GGUF_TYPE_INT8:
        case moe_GGUF_TYPE_BOOL:
            *out = 1;
            return 1;
        case moe_GGUF_TYPE_UINT16:
        case moe_GGUF_TYPE_INT16:
            *out = 2;
            return 1;
        case moe_GGUF_TYPE_UINT32:
        case moe_GGUF_TYPE_INT32:
        case moe_GGUF_TYPE_FLOAT32:
            *out = 4;
            return 1;
        case moe_GGUF_TYPE_UINT64:
        case moe_GGUF_TYPE_INT64:
        case moe_GGUF_TYPE_FLOAT64:
            *out = 8;
            return 1;
        default:
            return 0;
    }
}

static int moe_gguf_skip_string(std::ifstream& input) {
    uint64_t len = 0;
    return moe_gguf_read_u64(input, &len) && moe_gguf_skip_bytes(input, len);
}

static int moe_gguf_read_string(std::ifstream& input, std::string* out, uint64_t max_bytes) {
    uint64_t len = 0;
    if (!out || !moe_gguf_read_u64(input, &len)) {
        return 0;
    }
    if (len > max_bytes || len > (uint64_t)std::numeric_limits<size_t>::max()) {
        moe_gguf_skip_bytes(input, len);
        return 0;
    }
    out->assign((size_t)len, '\0');
    return len == 0 || moe_gguf_read_exact(input, &(*out)[0], (size_t)len);
}

static int moe_gguf_skip_value(std::ifstream& input, uint32_t type) {
    if (type == moe_GGUF_TYPE_STRING) {
        return moe_gguf_skip_string(input);
    }
    if (type == moe_GGUF_TYPE_ARRAY) {
        uint32_t item_type = 0;
        uint64_t item_count = 0;
        if (!moe_gguf_read_u32(input, &item_type) || !moe_gguf_read_u64(input, &item_count)) {
            return 0;
        }
        if (item_type == moe_GGUF_TYPE_STRING) {
            for (uint64_t i = 0; i < item_count; ++i) {
                if (!moe_gguf_skip_string(input)) {
                    return 0;
                }
            }
            return 1;
        }
        uint64_t item_size = 0;
        if (!moe_gguf_fixed_type_size(item_type, &item_size) ||
            item_count > UINT64_MAX / item_size) {
            return 0;
        }
        return moe_gguf_skip_bytes(input, item_count * item_size);
    }
    uint64_t size = 0;
    return moe_gguf_fixed_type_size(type, &size) && moe_gguf_skip_bytes(input, size);
}

static int moe_gguf_read_uint_value(std::ifstream& input, uint32_t type, uint64_t* out) {
    if (!out) {
        return 0;
    }
    switch (type) {
        case moe_GGUF_TYPE_UINT8: {
            uint8_t v = 0;
            if (!moe_gguf_read_u8(input, &v)) return 0;
            *out = v;
            return 1;
        }
        case moe_GGUF_TYPE_BOOL: {
            uint8_t v = 0;
            if (!moe_gguf_read_u8(input, &v)) return 0;
            *out = v ? 1u : 0u;
            return 1;
        }
        case moe_GGUF_TYPE_UINT16: {
            uint16_t v = 0;
            if (!moe_gguf_read_u16(input, &v)) return 0;
            *out = v;
            return 1;
        }
        case moe_GGUF_TYPE_UINT32: {
            uint32_t v = 0;
            if (!moe_gguf_read_u32(input, &v)) return 0;
            *out = v;
            return 1;
        }
        case moe_GGUF_TYPE_UINT64:
            return moe_gguf_read_u64(input, out);
        default:
            return 0;
    }
}

static int moe_gguf_read_float_value(std::ifstream& input, uint32_t type, double* out) {
    if (!out) {
        return 0;
    }
    switch (type) {
        case moe_GGUF_TYPE_FLOAT32: {
            float v = 0.0f;
            if (!moe_gguf_read_exact(input, &v, sizeof(v))) return 0;
            *out = (double)v;
            return 1;
        }
        case moe_GGUF_TYPE_FLOAT64: {
            double v = 0.0;
            if (!moe_gguf_read_exact(input, &v, sizeof(v))) return 0;
            *out = v;
            return 1;
        }
        case moe_GGUF_TYPE_UINT8:
        case moe_GGUF_TYPE_BOOL:
        case moe_GGUF_TYPE_UINT16:
        case moe_GGUF_TYPE_UINT32:
        case moe_GGUF_TYPE_UINT64: {
            uint64_t v = 0;
            if (!moe_gguf_read_uint_value(input, type, &v)) return 0;
            *out = (double)v;
            return 1;
        }
        default:
            return 0;
    }
}
