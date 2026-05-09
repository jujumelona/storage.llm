#pragma once
#include "candidate.h"
#include <vector>

namespace storagellm::autotune {
HostInfo detect_host();
std::vector<Candidate> make_backend_plan(const HostInfo& host);
}
