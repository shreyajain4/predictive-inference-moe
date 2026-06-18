// moe_warm_profile.h
// ------------------
// Warm-from-history loader for MoeExpertCache.
//
// Reads a flat-text profile produced by npy_to_warm_profile.py (each line:
// "<layer> <expert_id> <hits>"; lines starting with '#' are comments), and
// applies it to a MoeExpertCache by calling cache.pin() for each entry.
//
// The caller supplies a src_provider callback that returns the CPU-side
// pointer for one expert's worth of weights, just like the path that already
// pins shared experts. The callback returns nullptr to skip an entry (e.g.
// for a layer that isn't actually MoE in this build).
//
// Wiring sequence (in the bench/main code):
//
//   MoeExpertCache cache(n_slots, expert_bytes, prefetch_stream, /*protect=*/0);
//   pin_shared_experts(cache, model);  // existing path
//
//   auto entries = load_warm_profile("data/user_history/shreya_warm_k32.txt");
//   auto src_provider = [&model](int layer, int expert) -> const void * {
//       // same CPU pointer resolution as your shared-expert pinning uses
//       return get_expert_cpu_ptr(model, layer, expert);
//   };
//   apply_warm_profile(cache, entries, src_provider);
//
//   // begin decode

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

class MoeExpertCache;

struct MoeWarmEntry {
    int     layer;
    int     expert;
    int64_t hits;   // informational; not used by the cache
};

// Returns an empty vector and logs to stderr on open failure or malformed file.
std::vector<MoeWarmEntry> load_warm_profile(const std::string & path);

// Calls cache.pin(layer, expert, src_provider(layer, expert)) for each entry.
// Skips entries where src_provider returns nullptr, logging a warning.
//
// Returns the number of entries actually pinned. Logs final count to stderr.
//
// Note: pinned slots come out of the cache's n_slots budget. Size the cache
// for entries.size() + headroom for managed/LRU slots.
std::size_t apply_warm_profile(
    MoeExpertCache & cache,
    const std::vector<MoeWarmEntry> & entries,
    std::function<const void *(int layer, int expert)> src_provider);
