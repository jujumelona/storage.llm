#pragma once

#include "json_scan.h"
#include "manifest_lookup.h"

namespace storagellm {

bool parse_projection_blocks(const JsonSlice& proj, ProjectionBlocks* out);

}  // namespace storagellm
