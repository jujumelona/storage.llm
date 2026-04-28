
// Bug #26: O(1) hash map lookup replaces O(N) linear scan
const glm5_scale4_entry_t* glm5_scale4_find(const glm5_scale4_file_t* file, const char* key) {
    if (!file || !file->entries || !key) {
        return NULL;
    }
    // Use hash map if available (built at open time)
    if (file->key_map) {
        auto it = file->key_map->find(key);
        if (it != file->key_map->end()) {
            return &file->entries[it->second];
        }
        return NULL;
    }
    // Fallback: linear scan (only if hash map failed to allocate)
    const size_t key_len = strlen(key);
    for (uint32_t i = 0; i < file->header->entry_count; ++i) {
        const glm5_scale4_entry_t* entry = &file->entries[i];
        // Bug #25: memcpy for ARM-safe read of packed key_length
        uint32_t entry_key_len;
        memcpy(&entry_key_len, &entry->key_length, sizeof(entry_key_len));
        if ((size_t)entry_key_len != key_len) {
            continue;
        }
        const char* entry_key = glm5_scale4_entry_key(file, entry);
        if (entry_key && memcmp(entry_key, key, key_len) == 0) {
            return entry;
        }
    }
    return NULL;
}

