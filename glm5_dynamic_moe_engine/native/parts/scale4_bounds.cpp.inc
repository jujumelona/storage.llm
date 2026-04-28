static int glm5_scale4_bounds_ok(const glm5_scale4_file_t* file, uint64_t offset, uint64_t bytes) {
    if (!file || !file->data) {
        return 0;
    }
    if (offset > (uint64_t)file->size) {
        return 0;
    }
    if (bytes > (uint64_t)file->size - offset) {
        return 0;
    }
    return 1;
}

