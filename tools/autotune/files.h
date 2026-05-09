#pragma once
#include "candidate.h"
#include <string>
#include <vector>

namespace storagellm::autotune {
void ensure_dir(const std::string& path);
void write_text(const std::string& path, const std::string& text);
std::string read_text(const std::string& path);
std::string json_escape(const std::string& s);
void write_profile_json(const std::string& path, const Candidate& c);
void write_selected_env(const std::string& path, const Candidate* selected);
void write_report_json(const std::string& path, const HostInfo& host, const std::vector<Candidate>& candidates, const Candidate* selected);
}
