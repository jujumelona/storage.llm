#include "select.h"
#include <algorithm>

namespace storagellm::autotune {

Candidate* select_best(std::vector<Candidate>& candidates) {
    Candidate* best = nullptr;
    for (auto& c : candidates) {
        // Fail-closed: only real measured candidates are selected automatically.
        if (!c.loadable || !c.measured || c.latency_ms <= 0.0) continue;
        if (!best || c.latency_ms < best->latency_ms) best = &c;
    }
    return best;
}

} // namespace storagellm::autotune
