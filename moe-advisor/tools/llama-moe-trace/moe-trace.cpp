// llama-moe-trace
// ----------------
// For each query in a TSV file (one query per line, user_id<TAB>text), runs
// prefill through the model and dumps per-token MoE routing to stdout as CSV:
//
//   user_id,query_id,layer,token_pos,expert_0,expert_1,...,expert_{K-1}
//
// Routing is captured via params.cb_eval, listening for tensors named
// "ffn_moe_topk-<layer>". K is the model's n_expert_used (top-k routing width).
//
// Design intent: build a per-(layer, expert) frequency profile from a user's
// historical prompts, so a snap-cache can be warmed at session start.
//
// This is intentionally prefill-only — generated decode tokens are model
// output, not user-typed text, and would skew the profile.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

struct LayerTopk {
    std::vector<int32_t> indices;        // [n_tokens * n_expert_used]
    int64_t n_tokens      = 0;
    int64_t n_expert_used = 0;
};

struct moe_trace_ctx {
    std::string                user_id;
    int                        query_id        = 0;
    int                        n_expert_used   = 0;
    bool                       header_written  = false;
    std::vector<uint8_t>       tmp_buf;
    std::map<int, LayerTopk>   layer_bufs;     // layer_id -> buffered topk
};

static bool moe_trace_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) {
        return strncmp(t->name, "ffn_moe_topk-", 13) == 0;
    }
    auto * ctx = static_cast<moe_trace_ctx *>(user_data);

    int layer = -1;
    if (sscanf(t->name, "ffn_moe_topk-%d", &layer) != 1) return true;

    const int64_t n_expert_used = t->ne[0];
    const int64_t n_tokens      = t->ne[1];
    const size_t  n_bytes       = ggml_nbytes(t);
    ctx->tmp_buf.resize(n_bytes);
    ggml_backend_tensor_get(t, ctx->tmp_buf.data(), 0, n_bytes);

    LayerTopk & buf = ctx->layer_bufs[layer];
    buf.n_tokens      = n_tokens;
    buf.n_expert_used = n_expert_used;
    buf.indices.resize(n_tokens * n_expert_used);
    for (int64_t tok = 0; tok < n_tokens; ++tok) {
        for (int64_t k = 0; k < n_expert_used; ++k) {
            const size_t off = (size_t)(k * t->nb[0] + tok * t->nb[1]);
            int32_t v;
            std::memcpy(&v, ctx->tmp_buf.data() + off, sizeof(int32_t));
            buf.indices[tok * n_expert_used + k] = v;
        }
    }
    ctx->n_expert_used = (int)n_expert_used;
    return true;
}

static void emit_csv(moe_trace_ctx * ctx) {
    if (ctx->layer_bufs.empty()) return;
    if (!ctx->header_written) {
        std::printf("user_id,query_id,layer,token_pos");
        for (int k = 0; k < ctx->n_expert_used; ++k) std::printf(",expert_%d", k);
        std::printf("\n");
        ctx->header_written = true;
    }
    for (auto & kv : ctx->layer_bufs) {
        const int layer = kv.first;
        const LayerTopk & lt = kv.second;
        for (int64_t tok = 0; tok < lt.n_tokens; ++tok) {
            std::printf("%s,%d,%d,%lld",
                        ctx->user_id.c_str(), ctx->query_id, layer, (long long)tok);
            for (int64_t k = 0; k < lt.n_expert_used; ++k) {
                std::printf(",%d", lt.indices[tok * lt.n_expert_used + k]);
            }
            std::printf("\n");
        }
    }
    ctx->layer_bufs.clear();
}

struct query_entry {
    std::string user_id;
    std::string text;
};

static std::vector<query_entry> load_queries(const std::string & path) {
    std::vector<query_entry> queries;
    std::ifstream f(path);
    if (!f.is_open()) {
        LOG_ERR("cannot open query file: %s\n", path.c_str());
        return queries;
    }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        const auto sep = line.find('\t');
        if (sep == std::string::npos) queries.push_back({"unknown", line});
        else                          queries.push_back({line.substr(0, sep), line.substr(sep + 1)});
    }
    return queries;
}

int main(int argc, char ** argv) {
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    if (params.prompt_file.empty()) {
        LOG_ERR("Usage: llama-moe-trace -m model.gguf -f queries.tsv > routing.csv\n");
        LOG_ERR("  queries.tsv: one query per line, user_id<TAB>query_text\n");
        LOG_ERR("  CSV columns: user_id,query_id,layer,token_pos,expert_*\n");
        return 1;
    }

    moe_trace_ctx ctx;
    params.cb_eval           = moe_trace_cb;
    params.cb_eval_user_data = &ctx;
    params.warmup            = false;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    auto * model    = llama_init->model();
    auto * lctx     = llama_init->context();
    if (!model || !lctx) {
        LOG_ERR("failed to init model\n");
        return 1;
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const bool          add_bos = llama_vocab_get_add_bos(vocab);

    auto queries = load_queries(params.prompt_file);
    if (queries.empty()) {
        LOG_ERR("no queries loaded\n");
        return 1;
    }
    LOG_INF("loaded %zu queries\n", queries.size());

    for (size_t q = 0; q < queries.size(); ++q) {
        ctx.user_id  = queries[q].user_id;
        ctx.query_id = (int)q;
        ctx.layer_bufs.clear();

        llama_memory_clear(llama_get_memory(lctx), true);

        std::vector<llama_token> tokens =
            common_tokenize(vocab, queries[q].text, add_bos, true);
        if (tokens.empty()) continue;

        // Prefill in one shot.
        if (llama_decode(lctx, llama_batch_get_one(tokens.data(), tokens.size())) != 0) {
            LOG_ERR("decode failed for query %zu\n", q);
            continue;
        }
        emit_csv(&ctx);

        if ((q + 1) % 25 == 0 || q + 1 == queries.size()) {
            LOG_INF("processed %zu/%zu queries\n", q + 1, queries.size());
        }
    }

    llama_backend_free();
    return 0;
}
