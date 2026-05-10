#pragma once
#include "candidate.h"
#include <vector>

namespace storagellm::autotune {
Candidate* select_best(std::vector<Candidate>& candidates);
}
