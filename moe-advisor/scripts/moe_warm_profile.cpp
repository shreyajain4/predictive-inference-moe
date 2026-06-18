#include "moe_warm_profile.h"
#include "moe_expert_cache.h"

#include <cstdio>
#include <fstream>
#include <sstream>


std::vector<MoeWarmEntry> load_warm_profile(const std::string & path) {
    std::vector<MoeWarmEntry> entries;
    std::ifstream f(path);
    if (!f.is_open()) {
        std::fprintf(stderr, "[moe-warm-profile] cannot open %s\n", path.c_str());
        return entries;
    }
    std::string line;
    std::size_t bad = 0;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        MoeWarmEntry e{};
        if (!(iss >> e.layer >> e.expert >> e.hits)) {
            ++bad;
            continue;
        }
        if (e.layer < 0 || e.expert < 0) {
            ++bad;
            continue;
        }
        entries.push_back(e);
    }
    std::fprintf(stderr, "[moe-warm-profile] loaded %zu entries from %s (%zu malformed lines)\n",
                 entries.size(), path.c_str(), bad);
    return entries;
}

std::size_t apply_warm_profile(
    MoeExpertCache & cache,
    const std::vector<MoeWarmEntry> & entries,
    std::function<const void *(int, int)> src_provider) {
    std::size_t pinned = 0;
    std::size_t skipped_no_src = 0;
    for (const auto & e : entries) {
        const void * src = src_provider(e.layer, e.expert);
        if (!src) {
            ++skipped_no_src;
            continue;
        }
        cache.pin(e.layer, e.expert, src);
        ++pinned;
    }
    std::fprintf(stderr,
        "[moe-warm-profile] pinned %zu of %zu (%zu skipped: no src)\n",
        pinned, entries.size(), skipped_no_src);
    return pinned;
}
