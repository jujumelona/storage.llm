#pragma once

#include "juju_header.h"
#include "../engine_core/core/mmap_loader.h"

#include <cstdint>
#include <string>

namespace storagellm {

class JujuPart {
public:
    JujuPart() = default;
    ~JujuPart();
    bool open(const char* path);
    void close();
    bool slice(uint64_t offset, uint64_t length, const uint8_t** out) const;
    void prefetch(uint64_t offset, uint64_t length);
    const JujuHeader& header() const { return header_; }
    const std::string& path() const { return path_; }
    bool mapped() const { return mapped_; }

private:
    mmap_context_t map_{};
    JujuHeader header_{};
    std::string path_;
    bool mapped_ = false;
};

}  // namespace storagellm
