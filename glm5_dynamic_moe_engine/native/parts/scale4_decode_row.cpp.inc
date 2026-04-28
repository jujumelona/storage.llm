int glm5_scale4_decode_row(
    const glm5_scale4_file_t* file,
    const glm5_scale4_entry_t* entry,
    uint32_t row,
    float* out_scales,
    uint32_t out_count
) {
    if (!file || !entry || !out_scales || row >= entry->rows || out_count < entry->groups) {
        return 0;
    }
    for (uint32_t group = 0; group < entry->groups; ++group) {
        out_scales[group] = glm5_scale4_get_scale(file, entry, row, group);
    }
    return 1;
}

