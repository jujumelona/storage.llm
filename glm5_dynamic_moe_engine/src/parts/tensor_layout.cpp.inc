static uint8_t infer_weight_format(const glm5_tensor_info_t& info) {
    const uint64_t elems = (uint64_t)info.rows * info.cols;
    if (info.weight_byte_length == elems * 2ull) {
        return 1;
    }
    if (info.weight_byte_length == elems * 4ull) {
        return 2;
    }
    return 0;
}

static uint64_t infer_weight_row_bytes(const glm5_tensor_info_t& info, uint8_t format) {
    if (format == 1) {
        return (uint64_t)info.cols * 2ull;
    }
    if (format == 2) {
        return (uint64_t)info.cols * 4ull;
    }
    return ((uint64_t)info.cols + 1u) / 2u;
}

static uint8_t infer_scale_format(const glm5_tensor_info_t& info) {
    const uint64_t count = (uint64_t)info.rows * info.groups;
    if (count == 0 || info.scale_byte_length == 0) {
        return 0;
    }
    if (info.scale_byte_length == count * 4ull) {
        return 1;
    }
    if (info.scale_byte_length == count * 2ull) {
        return 2;
    }
    return info.scale_byte_length == count ? 3 : 0;
}

static void finalize_tensor_layout(glm5_tensor_record* rec) {
    if (!rec) {
        return;
    }
    rec->weight_format = infer_weight_format(rec->info);
    rec->weight_row_bytes = infer_weight_row_bytes(rec->info, rec->weight_format);
    rec->scale_format = infer_scale_format(rec->info);
}

static int row_bounds_ok(const glm5_tensor_record& rec, uint32_t row) {
    if (row >= rec.info.rows || rec.info.cols == 0 || rec.weight_row_bytes == 0) {
        return 0;
    }
    const uint64_t rel = (uint64_t)row * rec.weight_row_bytes;
    return rel + rec.weight_row_bytes <= rec.info.weight_byte_length;
}

static int tensor_weight_format(const glm5_tensor_record& rec) {
    return rec.weight_format;
}
