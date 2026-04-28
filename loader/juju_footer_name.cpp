#include "juju_footer.h"

#include <cstdio>

namespace storagellm {

std::string juju_footer_name(uint32_t part) {
    char name[32];
    std::snprintf(name, sizeof(name), "part%02u.footer.json", part);
    return std::string(name);
}

}  // namespace storagellm
