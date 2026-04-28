static float glm5_raw_scale_at(
    const glm5_tensor_record& rec,
    const mmap_context_t* shard,
    uint32_t row,
    uint32_t group
) {
    if (!shard || !mmap_get_ptr(shard, 0) || rec.info.scale_byte_length == 0 || rec.info.groups == 0) {
        return 1.0f;
    }
    const uint64_t idx = (uint64_t)row * rec.info.groups + group;
    const uint64_t count = (uint64_t)rec.info.rows * rec.info.groups;
    const uint64_t shard_size = (uint64_t)mmap_get_size(shard);
    if (idx >= count || rec.info.scale_byte_offset > shard_size ||
        rec.info.scale_byte_length > shard_size - rec.info.scale_byte_offset) {
        return 1.0f;
    }
    const uint8_t* base = static_cast<const uint8_t*>(mmap_get_ptr(shard, static_cast<size_t>(rec.info.scale_byte_offset)));
    if (!base) {
        return 1.0f;
    }
    if (rec.scale_format == 1) {
        float v;
        std::memcpy(&v, base + idx * 4ull, sizeof(v));
        return v;
    }
    if (rec.scale_format == 2) {
        uint16_t v;
        std::memcpy(&v, base + idx * 2ull, sizeof(v));
        return glm5_raw_fp16_to_f32(v);
    }
    if (rec.scale_format == 3) {
        return storage_f8_e4m3_to_f32(base[idx]);
    }
    return 1.0f;
}
