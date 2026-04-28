#include <cstdio>
#include <string>

namespace storagellm {

std::string manifest_expert_key(uint32_t layer, uint32_t expert) {
    char key[32];
    // Fix: Support layer/expert > 999 without silent collision
    // Use full integer format instead of clamping to 999
    std::snprintf(key, sizeof(key), "L%u.E%u", layer, expert);
    return key;
}

}  // namespace storagellm
