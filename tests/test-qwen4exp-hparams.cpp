// qwen4exp hparams from DwarfStar-packed metadata: u64 integer arrays, no M-RoPE sections.
// The files have no tensors, so a load that gets past the hparams still fails later;
// the log tells apart an hparams error from a later one.

#include "llama.h"
#include "gguf.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

static std::string g_log;

static void log_cb(ggml_log_level /*level*/, const char * text, void * /*user_data*/) {
    g_log += text;
}

static void write_model(const std::string & path, uint32_t n_rot) {
    gguf_context * ctx = gguf_init_empty();

    gguf_set_val_str(ctx, "general.architecture", "qwen4exp");
    gguf_set_val_u32(ctx, "qwen4exp.block_count",                          49);
    gguf_set_val_u32(ctx, "qwen4exp.nextn_predict_layers",                 1);
    gguf_set_val_u32(ctx, "qwen4exp.context_length",                       262144);
    gguf_set_val_u32(ctx, "qwen4exp.embedding_length",                     2560);
    gguf_set_val_u32(ctx, "qwen4exp.vocab_size",                           248320);
    gguf_set_val_u32(ctx, "qwen4exp.attention.head_count",                 24);
    gguf_set_val_u32(ctx, "qwen4exp.attention.head_count_kv",              2);
    gguf_set_val_u32(ctx, "qwen4exp.attention.key_length",                 256);
    gguf_set_val_u32(ctx, "qwen4exp.attention.value_length",               256);
    gguf_set_val_f32(ctx, "qwen4exp.attention.layer_norm_rms_epsilon",     1e-6f);
    gguf_set_val_u32(ctx, "qwen4exp.attention.indexer.head_count",         4);
    gguf_set_val_u32(ctx, "qwen4exp.attention.indexer.key_length",         128);
    gguf_set_val_u32(ctx, "qwen4exp.attention.indexer.top_k",              2048);
    gguf_set_val_u32(ctx, "qwen4exp.expert_count",                         512);
    gguf_set_val_u32(ctx, "qwen4exp.expert_used_count",                    10);
    gguf_set_val_u32(ctx, "qwen4exp.expert_feed_forward_length",           640);
    gguf_set_val_u32(ctx, "qwen4exp.expert_shared_feed_forward_length",    640);
    gguf_set_val_u32(ctx, "qwen4exp.ssm.conv_kernel",                      4);
    gguf_set_val_u32(ctx, "qwen4exp.ssm.state_size",                       128);
    gguf_set_val_u32(ctx, "qwen4exp.ssm.group_count",                      16);
    gguf_set_val_u32(ctx, "qwen4exp.ssm.time_step_rank",                   48);
    gguf_set_val_u32(ctx, "qwen4exp.ssm.inner_size",                       6144);
    gguf_set_val_u32(ctx, "qwen4exp.full_attention_interval",              4);
    gguf_set_val_u32(ctx, "qwen4exp.hyper_connection.count",               4);
    gguf_set_val_u32(ctx, "qwen4exp.hyper_connection.low_rank",            320);
    gguf_set_val_u32(ctx, "qwen4exp.rope.dimension_count",                 n_rot);
    gguf_set_val_f32(ctx, "qwen4exp.rope.freq_base",                       1e7f);

    // stored as u64, like DwarfStar does
    std::vector<uint64_t> compress_ratios(49, 0);
    for (size_t il = 3; il < compress_ratios.size(); il += 4) {
        compress_ratios[il] = 4;
    }
    compress_ratios.back() = 4;
    gguf_set_arr_data(ctx, "qwen4exp.attention.compress_ratios", GGUF_TYPE_UINT64, compress_ratios.data(), compress_ratios.size());

    gguf_set_val_str(ctx, "tokenizer.ggml.model", "none");

    GGML_ASSERT(gguf_write_to_file(ctx, path.c_str(), false));
    gguf_free(ctx);
}

// true when the load got past the hparams (it fails later: the file has no tensors)
static bool hparams_ok(const std::string & path, std::string & err) {
    g_log.clear();
    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(path.c_str(), mparams);
    if (model) {
        llama_model_free(model);
    }
    const size_t pos = g_log.find("error loading model hyperparameters");
    if (pos == std::string::npos) {
        return true;
    }
    err = g_log.substr(pos, g_log.find('\n', pos) - pos);
    return false;
}

int main() {
    llama_log_set(log_cb, nullptr);
    llama_backend_init();

    const std::string path = "test-qwen4exp-hparams.gguf";
    int n_fail = 0;

    struct test_case {
        const char * desc;
        uint32_t     n_rot;
        bool         expect_ok;
    };
    const test_case cases[] = {
        { "dwarfstar_metadata_loads",          64, true  },
        { "default_sections_reject_bad_n_rot", 48, false },
    };

    for (const test_case & tc : cases) {
        write_model(path, tc.n_rot);
        std::string err;
        const bool ok = hparams_ok(path, err) == tc.expect_ok &&
                        (tc.expect_ok || err.find("rope.dimension_sections") != std::string::npos);
        printf("%s: %s%s%s\n", tc.desc, ok ? "OK" : "FAIL", err.empty() ? "" : " - ", err.c_str());
        n_fail += ok ? 0 : 1;
    }

    std::remove(path.c_str());
    llama_backend_free();

    return n_fail == 0 ? 0 : 1;
}
