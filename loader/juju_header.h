#pragma once

#include <cstddef>
#include <cstdint>

namespace storagellm {

#pragma pack(push, 1)
struct JujuHeader {
    char magic[8]{};           // 8 bytes
    uint32_t version = 0;      // 4 bytes
    uint32_t flags = 0;        // 4 bytes
    uint64_t header_bytes = 0; // 8 bytes
    uint64_t data_offset = 0;  // 8 bytes
    uint64_t json_offset = 0;  // 8 bytes
    uint64_t json_bytes = 0;   // 8 bytes
    uint64_t file_size = 0;    // 8 bytes
    uint64_t block_count = 0;  // 8 bytes
};  // Total: 64 bytes
#pragma pack(pop)

static_assert(sizeof(JujuHeader) == 64, "JujuHeader must be 64 bytes for binary compatibility");

bool juju_parse_header(const void* data, size_t size, JujuHeader* out);
bool juju_header_valid(const JujuHeader& header, size_t mapped_size);

}  // namespace storagellm
