float glm5_scale4_get_scale(
    const glm5_scale4_file_t* file,
    const glm5_scale4_entry_t* entry,
    uint32_t row,
    uint32_t group
) {
    if (!file || !entry || row >= entry->rows || group >= entry->groups ||
        entry->bits != 4 || entry->centroids != 16) {
        return 0.0f;
    }

    const uint64_t element = (uint64_t)row * (uint64_t)entry->groups + (uint64_t)group;
    const uint64_t index_byte_offset = entry->index_offset + element / 2u;
    if (!glm5_scale4_bounds_ok(file, index_byte_offset, 1)) {
        return 0.0f;
    }
    const uint8_t packed = file->data[index_byte_offset];
    const uint8_t code = (element & 1u) ? (uint8_t)(packed & 0x0Fu) : (uint8_t)(packed >> 4);

    const uint64_t codebook_offset = entry->codebook_offset + (uint64_t)code * sizeof(float);
    if (!glm5_scale4_bounds_ok(file, codebook_offset, sizeof(float))) {
        return 0.0f;
    }
    float value = 0.0f;
    memcpy(&value, file->data + codebook_offset, sizeof(float));
    return value;
}

