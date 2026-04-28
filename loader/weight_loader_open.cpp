#include "weight_loader.h"
#include "path_join.h"

namespace storagellm {

bool WeightLoader::open(
    const char* model_root,
    const char* manifest_path,
    const char* footer_root
) {
    close();
    if (!manifest_path || !manifest_.load(manifest_path)) {
        return false;
    }
    model_root_ = model_root ? model_root : "";
    footer_root_ = footer_root ? footer_root : "";
    return true;
}

void WeightLoader::close() {
    part_.close();
    footer_ = JujuFooter{};
    model_root_.clear();
    footer_root_.clear();
    active_part_ = UINT32_MAX;
}

bool WeightLoader::open_part(uint32_t part, const std::string& rel_path) {
    if (active_part_ == part && part_.mapped() && !footer_.empty()) {
        return true;
    }
    const std::string part_path = path_join(model_root_, rel_path);
    const std::string footer_path = path_join(footer_root_, juju_footer_name(part));
    if (!part_.open(part_path.c_str())) {
        const size_t slash = rel_path.find_last_of("/\\");
        if (slash == std::string::npos || slash + 1 >= rel_path.size()) {
            return false;
        }
        const std::string flat_path = path_join(model_root_, rel_path.substr(slash + 1));
        if (flat_path == part_path || !part_.open(flat_path.c_str())) {
            return false;
        }
    }
    if (!footer_.load(footer_path.c_str())) {
        part_.close();
        return false;
    }
    active_part_ = part;
    return true;
}

}  // namespace storagellm
