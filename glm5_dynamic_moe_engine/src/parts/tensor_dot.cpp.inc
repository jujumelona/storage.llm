static int glm5_dot_tensor_record_row(
    glm5_pc_engine_t* engine,
    const glm5_tensor_record* rec,
    const mmap_context_t* shard,
    uint32_t row,
    const float* x,
    uint32_t x_count,
    float* out_value
) {
    if (!engine || !rec || !x || !out_value ||
        x_count != rec->info.cols || !row_bounds_ok(*rec, row)) {
        return 0;
    }
    if (!shard || !mmap_get_ptr(shard, 0)) {
        return 0;
    }
    const int weight_format = tensor_weight_format(*rec);
    const uint64_t row_offset = rec->info.weight_byte_offset + (uint64_t)row * rec->weight_row_bytes;
    const uint64_t shard_size = (uint64_t)mmap_get_size(shard);
    if (row_offset > shard_size || rec->weight_row_bytes > shard_size - row_offset) {
        return 0;
    }
    const uint8_t* packed = static_cast<const uint8_t*>(mmap_get_ptr(shard, static_cast<size_t>(row_offset)));
    if (!packed) {
        return 0;
    }
    if (weight_format == 1) {
        *out_value = glm5_dot_raw_fp16_row(packed, x, rec->info.cols);
        return 1;
    }
    if (weight_format == 2) {
        *out_value = glm5_dot_raw_fp32_row(packed, x, rec->info.cols);
        return 1;
    }
    *out_value = glm5_dot_fp4_row(*rec, shard, packed, x, engine, row);
    return 1;
}
