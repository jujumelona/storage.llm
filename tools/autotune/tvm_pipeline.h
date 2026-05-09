#pragma once
#include "candidate.h"
#include <vector>

namespace storagellm::autotune {
void run_tvm_pipeline(const HostInfo& host, std::vector<Candidate>& candidates, int trials);
}
