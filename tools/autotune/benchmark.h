#pragma once
#include "candidate.h"
#include <vector>

namespace storagellm::autotune {
void benchmark_candidates(std::vector<Candidate>& candidates);
}
