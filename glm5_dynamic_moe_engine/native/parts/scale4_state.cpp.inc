
// Bug #26: Add hash map for O(1) key lookup instead of O(N) linear scan
#include <unordered_map>
#include <string>
#include <new>

struct glm5_scale4_file_t {
    mmap_context_t mmap;
    const uint8_t* data;
    size_t size;
    const glm5_scale4_header_t* header;
    const glm5_scale4_entry_t* entries;
    // Bug #26: key → entry_index hash map (built once at open time)
    std::unordered_map<std::string, uint32_t>* key_map;
};
