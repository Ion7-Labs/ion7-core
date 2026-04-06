/*
 * ion7_bridge.cpp - Stable C shim between ion7-core (LuaJIT) and libllama.so.
 *
 * Copyright (C) 2026 Ion7 Project Contributors
 * SPDX-License-Identifier: MIT
 *
 * This file is part of ion7-core.
 *
 * ion7-core is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Affero General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ion7-core is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with ion7-core. If not, see <https://www.gnu.org/licenses/>.
 *
 * ──────────────────────────────────────────────────────────────────────────
 * Migrated from C to C++ (ion7-core v2.0) to enable libcommon integration.
 * The public API is unchanged — all symbols remain extern "C".
 *
 * New in v2.0 (requires libcommon):
 *   - ion7_chat_templates_*   : Jinja2 templates with enable_thinking support
 *   - ion7_reasoning_budget_* : hard token budget inside <think> blocks
 *   - ion7_opt_*              : training / fine-tuning via llama_opt API
 * ──────────────────────────────────────────────────────────────────────────
 */

#include "ion7_bridge.h"

/* llama.cpp public API */
#include "llama.h"
#include "ggml-cpu.h"

/* libcommon — C++ layer */
#include "chat.h"
#include "common.h"
#include "reasoning-budget.h"
#include "sampling.h"
#include "speculative.h"
#include "regex-partial.h"
#include "json-schema-to-grammar.h"
#include "base64.hpp"

/* nlohmann/json — full implementation needed for ordered_json::parse() */
#include "nlohmann/json.hpp"

/* Standard C++ */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <sstream>
#include <string>
#include <vector>

/* =========================================================================
 * Version & info
 * ======================================================================= */

const char* ion7_bridge_version(void) { return "2.0.0"; }
const char* ion7_llama_info(void)     { return llama_print_system_info(); }

/* =========================================================================
 * Log
 * ======================================================================= */

static int g_log_level = 1;

static void ion7_log_cb(enum ggml_log_level level, const char* text, void* ud)
{
    (void)ud;
    if (g_log_level <= 0) return;
    int threshold = 5 - g_log_level;
    if ((int)level >= threshold) fputs(text, stderr);
}

/* =========================================================================
 * Backend
 * ======================================================================= */

void ion7_backend_init(void) { llama_log_set(ion7_log_cb, NULL); llama_backend_init(); }
void ion7_backend_free(void) { llama_backend_free(); }
void ion7_set_log_level(int l) { g_log_level = l; }

/* =========================================================================
 * Capabilities
 * ======================================================================= */

int ion7_supports_mmap          (void) { return llama_supports_mmap()          ? 1 : 0; }
int ion7_supports_mlock         (void) { return llama_supports_mlock()         ? 1 : 0; }
int ion7_supports_gpu_offload   (void) { return llama_supports_gpu_offload()   ? 1 : 0; }
int ion7_supports_rpc           (void) { return llama_supports_rpc()           ? 1 : 0; }
int ion7_max_devices            (void) { return (int)llama_max_devices(); }
int ion7_max_parallel_sequences (void) { return (int)llama_max_parallel_sequences(); }

/* =========================================================================
 * Model
 * ======================================================================= */

struct llama_model* ion7_model_load(const char* path, int32_t n_gpu_layers,
                                    int use_mmap, int use_mlock, int vocab_only)
{
    struct llama_model_params p = llama_model_default_params();
    p.n_gpu_layers = n_gpu_layers;
    p.use_mmap     = (bool)use_mmap;
    p.use_mlock    = (bool)use_mlock;
    p.vocab_only   = (bool)vocab_only;
    return llama_model_load_from_file(path, p);
}

void ion7_model_free(struct llama_model* m) { if (m) llama_model_free(m); }

struct llama_model* ion7_model_load_splits(const char** paths, size_t n_paths,
                                            int32_t n_gpu_layers)
{
    struct llama_model_params p = llama_model_default_params();
    p.n_gpu_layers = n_gpu_layers;
    return llama_model_load_from_splits(paths, n_paths, p);
}

int ion7_model_desc(const struct llama_model* m, char* buf, size_t sz)
{
    return llama_model_desc(m, buf, sz);
}

uint64_t ion7_model_size    (const struct llama_model* m) { return llama_model_size(m); }
uint64_t ion7_model_n_params(const struct llama_model* m) { return llama_model_n_params(m); }

int ion7_model_meta_val(const struct llama_model* m, const char* key,
                        char* buf, size_t sz)
{
    return llama_model_meta_val_str(m, key, buf, sz);
}

int ion7_model_meta_count(const struct llama_model* m)
{
    return llama_model_meta_count(m);
}

int ion7_model_meta_key_at(const struct llama_model* m, int32_t idx,
                            char* buf, size_t sz)
{
    return llama_model_meta_key_by_index(m, idx, buf, sz);
}

int ion7_model_meta_val_at(const struct llama_model* m, int32_t idx,
                            char* buf, size_t sz)
{
    return llama_model_meta_val_str_by_index(m, idx, buf, sz);
}

const char* ion7_model_chat_template(const struct llama_model* m, const char* name)
{
    return llama_model_chat_template(m, name);
}

int ion7_model_has_encoder (const struct llama_model* m) { return llama_model_has_encoder(m) ? 1 : 0; }
int ion7_model_has_decoder (const struct llama_model* m) { return llama_model_has_decoder(m) ? 1 : 0; }
int ion7_model_is_recurrent(const struct llama_model* m) { return llama_model_is_recurrent(m) ? 1 : 0; }
int ion7_model_is_hybrid   (const struct llama_model* m) { return llama_model_is_hybrid(m)    ? 1 : 0; }
int ion7_model_is_diffusion(const struct llama_model* m) { return llama_model_is_diffusion(m) ? 1 : 0; }

int32_t ion7_model_n_ctx_train(const struct llama_model* m) { return llama_model_n_ctx_train(m); }
int32_t ion7_model_n_embd    (const struct llama_model* m) { return llama_model_n_embd(m);     }
int32_t ion7_model_n_embd_inp(const struct llama_model* m) { return llama_model_n_embd_inp(m); }
int32_t ion7_model_n_embd_out(const struct llama_model* m) { return llama_model_n_embd_out(m); }
int32_t ion7_model_n_layer   (const struct llama_model* m) { return llama_model_n_layer(m);    }
int32_t ion7_model_n_head    (const struct llama_model* m) { return llama_model_n_head(m);     }
int32_t ion7_model_n_head_kv (const struct llama_model* m) { return llama_model_n_head_kv(m);  }
int32_t ion7_model_n_swa     (const struct llama_model* m) { return llama_model_n_swa(m);      }
int32_t ion7_model_n_cls_out (const struct llama_model* m) { return (int32_t)llama_model_n_cls_out(m); }
float   ion7_model_rope_freq_scale_train(const struct llama_model* m) { return llama_model_rope_freq_scale_train(m); }
int32_t ion7_model_decoder_start_token(const struct llama_model* m) { return (int32_t)llama_model_decoder_start_token(m); }

const char* ion7_model_cls_label(const struct llama_model* m, uint32_t i)
{
    return llama_model_cls_label(m, i);
}

const char* ion7_model_rope_type(const struct llama_model* m)
{
    switch (llama_model_rope_type(m)) {
        case LLAMA_ROPE_TYPE_NONE:   return "none";
        case LLAMA_ROPE_TYPE_NORM:   return "norm";
        case LLAMA_ROPE_TYPE_NEOX:   return "neox";
        case LLAMA_ROPE_TYPE_MROPE:  return "mrope";
        case LLAMA_ROPE_TYPE_IMROPE: return "imrope";
        case LLAMA_ROPE_TYPE_VISION: return "vision";
        default:                     return "unknown";
    }
}

void ion7_model_save(const struct llama_model* m, const char* path)
{
    llama_model_save_to_file(m, path);
}

int ion7_params_fit(const char* path, int32_t* n_gpu_layers,
                    uint32_t* n_ctx, uint32_t n_ctx_min)
{
    struct llama_model_params   mparams = llama_model_default_params();
    struct llama_context_params cparams = llama_context_default_params();

    cparams.n_ctx = *n_ctx;

    size_t max_dev = llama_max_devices();
    size_t max_ovr = llama_max_tensor_buft_overrides();
    float*                                   ts      = (float*)calloc(max_dev, sizeof(float));
    struct llama_model_tensor_buft_override* ov      = (struct llama_model_tensor_buft_override*)calloc(max_ovr,
        sizeof(struct llama_model_tensor_buft_override));
    size_t*                                  margins = (size_t*)calloc(max_dev, sizeof(size_t));

    enum llama_params_fit_status status = llama_params_fit(
        path, &mparams, &cparams,
        ts, ov, margins,
        n_ctx_min,
        GGML_LOG_LEVEL_WARN
    );

    if (status == LLAMA_PARAMS_FIT_STATUS_SUCCESS) {
        *n_gpu_layers = mparams.n_gpu_layers;
        *n_ctx        = cparams.n_ctx;
    }

    free(ts); free(ov); free(margins);
    return (int)status;
}

/* =========================================================================
 * Context
 * ======================================================================= */

struct llama_context* ion7_context_create(
    struct llama_model* model,
    uint32_t n_ctx, uint32_t n_batch, uint32_t n_ubatch,
    uint32_t n_seq_max,
    int32_t n_threads, int32_t n_threads_batch,
    int flash_attn, int offload_kqv, int op_offload,
    int no_perf, int type_k, int type_v, int swa_full, int kv_unified)
{
    struct llama_context_params p = llama_context_default_params();
    p.n_ctx           = n_ctx    ? n_ctx    : 4096;
    p.n_batch         = n_batch  ? n_batch  : 2048;
    p.n_ubatch        = n_ubatch ? n_ubatch : p.n_batch;
    p.n_seq_max       = n_seq_max ? n_seq_max : 1;
    p.n_threads       = n_threads > 0       ? n_threads       : 4;
    p.n_threads_batch = n_threads_batch > 0 ? n_threads_batch : p.n_threads * 2;
    p.flash_attn_type = (type_k != 1 || flash_attn)
                        ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                        : LLAMA_FLASH_ATTN_TYPE_AUTO;
    p.offload_kqv  = (bool)offload_kqv;
    p.op_offload   = (bool)op_offload;
    p.no_perf      = (bool)no_perf;
    p.swa_full     = (bool)swa_full;
    p.kv_unified   = (bool)kv_unified;

    if (type_k > 0) p.type_k = (enum ggml_type)type_k;
    if (type_v > 0) p.type_v = (enum ggml_type)type_v;
    return llama_init_from_model(model, p);
}

struct llama_context* ion7_embedding_context_create(
    struct llama_model* model, uint32_t n_ctx, uint32_t n_seq_max,
    int32_t n_threads, int pooling)
{
    struct llama_context_params p = llama_context_default_params();
    p.n_ctx           = n_ctx ? n_ctx : 512;
    p.n_batch         = p.n_ctx;
    p.n_ubatch        = p.n_batch;
    p.n_seq_max       = n_seq_max ? n_seq_max : 1;
    p.n_threads       = n_threads > 0 ? n_threads : 4;
    p.n_threads_batch = p.n_threads;
    p.embeddings      = true;
    p.pooling_type    = (enum llama_pooling_type)pooling;
    p.offload_kqv     = false;
    p.op_offload      = false;
    p.no_perf         = true;
    p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    return llama_init_from_model(model, p);
}

void ion7_context_free(struct llama_context* ctx) { if (ctx) llama_free(ctx); }

/* =========================================================================
 * KV cache
 * ======================================================================= */

static inline llama_memory_t ctx_mem(struct llama_context* ctx)
{
    return llama_get_memory(ctx);
}

void ion7_kv_clear(struct llama_context* ctx)
{
    llama_memory_t m = ctx_mem(ctx);
    if (m) llama_memory_clear(m, true);
}
void ion7_kv_seq_rm(struct llama_context* ctx, int32_t seq, int32_t p0, int32_t p1)
{
    llama_memory_t m = ctx_mem(ctx);
    if (m) llama_memory_seq_rm(m, seq, p0, p1);
}
void ion7_kv_seq_cp(struct llama_context* ctx, int32_t src, int32_t dst,
                    int32_t p0, int32_t p1)
{
    llama_memory_t m = ctx_mem(ctx);
    if (m) llama_memory_seq_cp(m, src, dst, p0, p1);
}
void ion7_kv_seq_keep(struct llama_context* ctx, int32_t seq)
{
    llama_memory_t m = ctx_mem(ctx);
    if (m) llama_memory_seq_keep(m, seq);
}
void ion7_kv_seq_shift(struct llama_context* ctx, int32_t seq,
                       int32_t p0, int32_t p1, int32_t delta)
{
    llama_memory_t m = ctx_mem(ctx);
    if (m) llama_memory_seq_add(m, seq, p0, p1, delta);
}

/* =========================================================================
 * State persistence
 * ======================================================================= */

size_t ion7_state_size(struct llama_context* ctx) { return llama_state_get_size(ctx); }

size_t ion7_state_get(struct llama_context* ctx, uint8_t* buf, size_t sz)
{
    return llama_state_get_data(ctx, buf, sz);
}
int ion7_state_set(struct llama_context* ctx, const uint8_t* buf, size_t sz)
{
    return (llama_state_set_data(ctx, buf, sz) > 0) ? 1 : 0;
}
int ion7_state_save_file(struct llama_context* ctx, const char* path,
                         const llama_token* tokens, size_t n)
{
    return llama_state_save_file(ctx, path, tokens, n) ? 1 : 0;
}
int ion7_state_load_file(struct llama_context* ctx, const char* path,
                         llama_token* out, size_t cap, size_t* n_out)
{
    return llama_state_load_file(ctx, path, out, cap, n_out) ? 1 : 0;
}

/* =========================================================================
 * LoRA adapters
 * ======================================================================= */

struct llama_adapter_lora* ion7_lora_load(struct llama_model* m, const char* path)
{
    return llama_adapter_lora_init(m, path);
}
void ion7_lora_free(struct llama_adapter_lora* a) { if (a) llama_adapter_lora_free(a); }

int ion7_lora_apply(struct llama_context* ctx, struct llama_adapter_lora* a, float scale)
{
    struct llama_adapter_lora* adapters[1] = { a };
    float scales[1] = { scale };
    return (int)llama_set_adapters_lora(ctx, adapters, 1, scales);
}
int ion7_lora_remove(struct llama_context* ctx)
{
    return (int)llama_set_adapters_lora(ctx, NULL, 0, NULL);
}
int ion7_lora_meta_val(const struct llama_adapter_lora* a, const char* key,
                       char* buf, size_t sz)
{
    return llama_adapter_meta_val_str(a, key, buf, sz);
}

/* =========================================================================
 * Performance
 * ======================================================================= */

void ion7_perf_print(struct llama_context* ctx) { llama_perf_context_print(ctx); }
void ion7_perf_reset(struct llama_context* ctx) { llama_perf_context_reset(ctx); }

void ion7_perf_get(struct llama_context* ctx,
                    double*  t_load_ms,   double*  t_p_eval_ms,
                    double*  t_eval_ms,   int32_t* n_p_eval,
                    int32_t* n_eval,      int32_t* n_reused)
{
    struct llama_perf_context_data d = llama_perf_context(ctx);
    if (t_load_ms)   *t_load_ms   = d.t_load_ms;
    if (t_p_eval_ms) *t_p_eval_ms = d.t_p_eval_ms;
    if (t_eval_ms)   *t_eval_ms   = d.t_eval_ms;
    if (n_p_eval)    *n_p_eval    = d.n_p_eval;
    if (n_eval)      *n_eval      = d.n_eval;
    if (n_reused)    *n_reused    = d.n_reused;
}

/* =========================================================================
 * Per-sequence state
 * ======================================================================= */

size_t ion7_state_seq_size(struct llama_context* ctx, int32_t seq_id)
{
    return llama_state_seq_get_size(ctx, (llama_seq_id)seq_id);
}

int ion7_state_seq_save(struct llama_context* ctx, const char* path, int32_t seq_id)
{
    size_t written = llama_state_seq_save_file(ctx, path, (llama_seq_id)seq_id, NULL, 0);
    return written > 0 ? 1 : 0;
}

int ion7_state_seq_load(struct llama_context* ctx, const char* path, int32_t dest_seq_id)
{
    size_t n_tokens_out = 0;
    size_t read = llama_state_seq_load_file(ctx, path, (llama_seq_id)dest_seq_id,
                                             NULL, 0, &n_tokens_out);
    return read > 0 ? 1 : 0;
}

/* =========================================================================
 * Threadpool
 * ======================================================================= */

ion7_threadpool_t* ion7_threadpool_create(int n_threads)
{
    struct ggml_threadpool_params p = ggml_threadpool_params_default(n_threads);
    return (ion7_threadpool_t*)ggml_threadpool_new(&p);
}

void ion7_threadpool_free(ion7_threadpool_t* tp)
{
    if (tp) ggml_threadpool_free((ggml_threadpool_t)tp);
}

void ion7_threadpool_attach(struct llama_context* ctx,
                             ion7_threadpool_t* tp,
                             ion7_threadpool_t* tp_batch)
{
    llama_attach_threadpool(ctx,
        (ggml_threadpool_t)tp,
        tp_batch ? (ggml_threadpool_t)tp_batch : (ggml_threadpool_t)tp);
}

void ion7_threadpool_detach(struct llama_context* ctx)
{
    llama_detach_threadpool(ctx);
}

int ion7_threadpool_n_threads(ion7_threadpool_t* tp)
{
    (void)tp;
    return 0;
}

void ion7_threadpool_pause(ion7_threadpool_t* tp)
{
    if (tp) ggml_threadpool_pause((ggml_threadpool_t)tp);
}

void ion7_threadpool_resume(ion7_threadpool_t* tp)
{
    if (tp) ggml_threadpool_resume((ggml_threadpool_t)tp);
}

/* =========================================================================
 * Model load from file descriptor
 * ======================================================================= */

struct llama_model* ion7_model_load_fd(int fd, int n_gpu_layers)
{
    FILE* f = fdopen(fd, "rb");
    if (!f) return NULL;

    struct llama_model_params p = llama_model_default_params();
    p.n_gpu_layers = n_gpu_layers;

    struct llama_model* model = llama_model_load_from_file_ptr(f, p);
    return model;
}

/* =========================================================================
 * Diagnostics
 * ======================================================================= */

void ion7_print_struct_sizes(void)
{
    fprintf(stderr, "[ion7-core] bridge               = %s\n", ion7_bridge_version());
    fprintf(stderr, "[ion7-core] llama_batch          = %zu\n", sizeof(struct llama_batch));
    fprintf(stderr, "[ion7-core] llama_context_params = %zu\n", sizeof(struct llama_context_params));
    fprintf(stderr, "[ion7-core] llama_model_params   = %zu\n", sizeof(struct llama_model_params));
    fprintf(stderr, "[ion7-core] llama_token          = %zu\n", sizeof(llama_token));
    fprintf(stderr, "[ion7-core] gpu_offload          = %s\n",  llama_supports_gpu_offload() ? "yes" : "no");
}

/* =========================================================================
 * Custom sampler trampolines
 * ======================================================================= */

typedef struct {
    char                   name[64];
    ion7_sampler_apply_fn  apply_fn;
    ion7_sampler_accept_fn accept_fn;
    ion7_sampler_reset_fn  reset_fn;
    ion7_sampler_free_fn   free_fn;
    void*                  userdata;
} ion7_custom_sampler_ctx_t;

static const char* _ion7_smpl_name(const struct llama_sampler* smpl) {
    return ((ion7_custom_sampler_ctx_t*)smpl->ctx)->name;
}
static void _ion7_smpl_accept(struct llama_sampler* smpl, llama_token tok) {
    ion7_custom_sampler_ctx_t* c = (ion7_custom_sampler_ctx_t*)smpl->ctx;
    if (c->accept_fn) c->accept_fn(tok, c->userdata);
}
static void _ion7_smpl_apply(struct llama_sampler* smpl, llama_token_data_array* cur_p) {
    ion7_custom_sampler_ctx_t* c = (ion7_custom_sampler_ctx_t*)smpl->ctx;
    if (c->apply_fn) c->apply_fn(cur_p, c->userdata);
}
static void _ion7_smpl_reset(struct llama_sampler* smpl) {
    ion7_custom_sampler_ctx_t* c = (ion7_custom_sampler_ctx_t*)smpl->ctx;
    if (c->reset_fn) c->reset_fn(c->userdata);
}
static void _ion7_smpl_free(struct llama_sampler* smpl) {
    ion7_custom_sampler_ctx_t* c = (ion7_custom_sampler_ctx_t*)smpl->ctx;
    if (c->free_fn) c->free_fn(c->userdata);
    free(c);
}
static struct llama_sampler* _ion7_smpl_clone(const struct llama_sampler* smpl) {
    ion7_custom_sampler_ctx_t* src = (ion7_custom_sampler_ctx_t*)smpl->ctx;
    return ion7_sampler_create(
        src->name, src->apply_fn, src->accept_fn,
        src->reset_fn, src->free_fn, src->userdata);
}

/* Field order matches struct llama_sampler_i exactly:
 * name, accept, apply, reset, clone, free, backend_init,
 * backend_accept, backend_apply, backend_set_input          */
static struct llama_sampler_i _ion7_custom_iface = {
    _ion7_smpl_name,    /* name   */
    _ion7_smpl_accept,  /* accept */
    _ion7_smpl_apply,   /* apply  */
    _ion7_smpl_reset,   /* reset  */
    _ion7_smpl_clone,   /* clone  */
    _ion7_smpl_free,    /* free   */
    nullptr,            /* backend_init      */
    nullptr,            /* backend_accept    */
    nullptr,            /* backend_apply     */
    nullptr,            /* backend_set_input */
};

struct llama_sampler* ion7_sampler_create(
    const char*            name,
    ion7_sampler_apply_fn  apply_fn,
    ion7_sampler_accept_fn accept_fn,
    ion7_sampler_reset_fn  reset_fn,
    ion7_sampler_free_fn   free_fn,
    void*                  userdata)
{
    ion7_custom_sampler_ctx_t* ctx =
        (ion7_custom_sampler_ctx_t*)malloc(sizeof(ion7_custom_sampler_ctx_t));
    if (!ctx) return nullptr;
    strncpy(ctx->name, name ? name : "custom", sizeof(ctx->name) - 1);
    ctx->name[sizeof(ctx->name) - 1] = '\0';
    ctx->apply_fn  = apply_fn;
    ctx->accept_fn = accept_fn;
    ctx->reset_fn  = reset_fn;
    ctx->free_fn   = free_fn;
    ctx->userdata  = userdata;
    return llama_sampler_init(&_ion7_custom_iface, (llama_sampler_context_t)ctx);
}

/* =========================================================================
 * Model quantization
 * ======================================================================= */

int ion7_model_quantize(
    const char* path_in, const char* path_out,
    int ftype, int n_threads, int pure, int allow_req, int dry_run)
{
    llama_model_quantize_params p = llama_model_quantize_default_params();
    p.ftype            = (enum llama_ftype)ftype;
    p.nthread          = n_threads;
    p.pure             = (bool)pure;
    p.allow_requantize = (bool)allow_req;
    p.dry_run          = (bool)dry_run;
    return (int)llama_model_quantize(path_in, path_out, &p);
}

/* =========================================================================
 * ── C++ EXTENSIONS (libcommon) ───────────────────────────────────────────
 * All functions below require libcommon and are new in bridge v2.0.
 * They are still exported as extern "C" (declared in ion7_bridge.h).
 * ======================================================================= */

/* ── Chat Templates (Jinja2 native, enable_thinking support) ───────────── */

ion7_chat_templates_t* ion7_chat_templates_init(
    const struct llama_model* model,
    const char* tmpl_override)
{
    auto ptr = common_chat_templates_init(model, tmpl_override ? tmpl_override : "");
    return (ion7_chat_templates_t*)ptr.release();
}

void ion7_chat_templates_free(ion7_chat_templates_t* t)
{
    if (t) common_chat_templates_free((common_chat_templates*)t);
}

int ion7_chat_templates_support_thinking(const ion7_chat_templates_t* t)
{
    if (!t) return 0;
    return common_chat_templates_support_enable_thinking(
        (const common_chat_templates*)t) ? 1 : 0;
}

/*
 * ion7_chat_templates_apply — apply a Jinja2 template with advanced options.
 *
 * roles[]    : array of n_msgs role strings ("system", "user", "assistant")
 * contents[] : array of n_msgs content strings
 * add_ass    : 1 = append assistant generation prefix
 * enable_thinking : -1 = model default, 0 = force off, 1 = force on
 *
 * Returns total bytes needed. If > buf_len, resize buf and call again.
 */
int32_t ion7_chat_templates_apply(
    ion7_chat_templates_t* t,
    const char** roles,
    const char** contents,
    size_t       n_msgs,
    int          add_ass,
    int          enable_thinking,
    char*        buf,
    int32_t      buf_len)
{
    if (!t) return -1;

    common_chat_templates_inputs inputs;
    inputs.add_generation_prompt = (bool)add_ass;
    inputs.use_jinja             = true;

    if (enable_thinking == 0)       inputs.enable_thinking = false;
    else if (enable_thinking == 1)  inputs.enable_thinking = true;
    /* -1 = leave at default (true) */

    for (size_t i = 0; i < n_msgs; i++) {
        common_chat_msg msg;
        msg.role    = roles[i]    ? roles[i]    : "";
        msg.content = contents[i] ? contents[i] : "";
        inputs.messages.push_back(std::move(msg));
    }

    common_chat_params result =
        common_chat_templates_apply((const common_chat_templates*)t, inputs);

    int32_t needed = (int32_t)result.prompt.size() + 1;
    if (buf && buf_len >= needed) {
        memcpy(buf, result.prompt.c_str(), needed);
    }
    return needed;
}

/* ── Reasoning Budget ──────────────────────────────────────────────────── */

/*
 * ion7_reasoning_budget_init — create a sampler that hard-limits the number
 * of tokens generated inside a <think> block.
 *
 * model   : used to tokenize the special tokens
 * n_budget: max tokens inside the block (0 = disable thinking entirely)
 *
 * Returns a llama_sampler* ready to be inserted into a sampler chain.
 * The chain takes ownership; do not free manually after adding.
 */
struct llama_sampler* ion7_reasoning_budget_init(
    const struct llama_model* model,
    int32_t n_budget)
{
    const llama_vocab* vocab = llama_model_get_vocab(model);

    /* Tokenize the Qwen3/3.5 thinking delimiters.
     * <think>\n  →  start of thinking block
     * \n</think>\n  →  end of thinking block
     * \n</think>\n  →  forced close token when budget is exceeded
     * <think>\n\n</think>\n  →  prefill for enable_thinking=false  */
    auto tokenize_str = [&](const std::string& s) -> std::vector<llama_token> {
        std::vector<llama_token> toks(s.size() + 8);
        int n = llama_tokenize(vocab, s.c_str(), (int32_t)s.size(),
                               toks.data(), (int32_t)toks.size(),
                               false, true);
        if (n < 0) return {};
        toks.resize(n);
        return toks;
    };

    auto start_toks   = tokenize_str("<think>\n");
    auto end_toks     = tokenize_str("\n</think>\n");
    auto forced_toks  = tokenize_str("\n</think>\n");
    auto prefill_toks = tokenize_str("<think>\n\n</think>\n");

    return common_reasoning_budget_init(
        vocab,
        start_toks,
        end_toks,
        forced_toks,
        n_budget,
        prefill_toks);
}

/* ── Training (llama_opt API) ──────────────────────────────────────────── */

/* Internal: simple fixed learning rate for opt param callback */
struct ion7_lr_params { float lr; };

static ggml_opt_optimizer_params _ion7_lr_callback(void* ud)
{
    ion7_lr_params* p = (ion7_lr_params*)ud;
    ggml_opt_optimizer_params op = ggml_opt_get_default_optimizer_params(ud);
    op.adamw.alpha = p->lr;
    return op;
}

/*
 * ion7_opt_init — initialise training on a context.
 *
 * optimizer : 0 = AdamW (recommended), 1 = SGD
 * lr        : learning rate (e.g. 1e-4 for LoRA, 1e-5 for full fine-tune)
 *
 * Must be called before ion7_opt_epoch.
 * The context must have been created without quantized KV cache (F16/F32).
 */
ion7_opt_state_t* ion7_opt_init(
    struct llama_context* ctx,
    struct llama_model*   model,
    int                   optimizer,
    float                 lr)
{
    ion7_lr_params* lrp = new ion7_lr_params{ lr };

    struct llama_opt_params p;
    p.n_ctx_train    = 0;  /* use context size */
    p.param_filter   = llama_opt_param_filter_all;
    p.param_filter_ud = nullptr;
    p.get_opt_pars   = _ion7_lr_callback;
    p.get_opt_pars_ud = (void*)lrp;
    p.optimizer_type = (optimizer == 1)
                       ? GGML_OPT_OPTIMIZER_TYPE_SGD
                       : GGML_OPT_OPTIMIZER_TYPE_ADAMW;

    llama_opt_init(ctx, model, p);
    return (ion7_opt_state_t*)lrp;  /* opaque handle to release later */
}

void ion7_opt_free(ion7_opt_state_t* state)
{
    delete (ion7_lr_params*)state;
}

/*
 * ion7_opt_dataset_create — build a training dataset from a token array.
 *
 * tokens   : flat array of llama_token (the full corpus, pre-tokenised)
 * n_tokens : number of tokens
 * n_ctx    : context window size (stride between samples)
 *
 * Returns an opaque dataset handle. Free with ion7_opt_dataset_free.
 */
ggml_opt_dataset_t ion7_opt_dataset_create(
    struct llama_context* ctx,
    const llama_token*    tokens,
    int64_t               n_tokens,
    int64_t               n_ctx)
{
    std::vector<llama_token> tok_vec(tokens, tokens + n_tokens);
    return common_opt_dataset_init(ctx, tok_vec, n_ctx);
}

void ion7_opt_dataset_free(ggml_opt_dataset_t dataset)
{
    if (dataset) ggml_opt_dataset_free(dataset);
}

/*
 * ion7_opt_epoch — run one training epoch.
 *
 * val_split : fraction of dataset used for validation (0.0 = no validation)
 * Returns   : training loss for this epoch, or -1.0 on error.
 */
float ion7_opt_epoch(
    struct llama_context* ctx,
    ggml_opt_dataset_t    dataset,
    float                 val_split)
{
    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_eval  = ggml_opt_result_init();

    int64_t n_data       = ggml_opt_dataset_data(dataset)->ne[1];
    int64_t idata_split  = (int64_t)(n_data * (1.0f - val_split));

    llama_opt_epoch(ctx, dataset,
                    result_train, result_eval,
                    idata_split,
                    nullptr, nullptr);

    double loss = 0.0, uncertainty = 0.0;
    ggml_opt_result_loss(result_train, &loss, &uncertainty);

    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_eval);

    return (float)loss;
}

/* =========================================================================
 * ── Context warmup ────────────────────────────────────────────────────────
 * ======================================================================= */

void ion7_context_warmup(struct llama_context* ctx)
{
    if (!ctx) return;
    const llama_model* model = llama_get_model(ctx);
    const llama_vocab* vocab = llama_model_get_vocab(model);
    llama_token dummy = llama_vocab_bos(vocab);
    if (dummy < 0) dummy = 0;
    struct llama_batch batch = llama_batch_get_one(&dummy, 1);
    llama_decode(ctx, batch);
    ion7_kv_clear(ctx);
}

/* =========================================================================
 * ── common_sampler (DRY, XTC, grammar_lazy, mirostat, logit bias) ────────
 * ======================================================================= */

/* Internal wrapper so we can store params alongside the sampler */
struct ion7_csampler {
    common_sampler* smpl;
};

ion7_csampler_t* ion7_csampler_init(
    const struct llama_model*     model,
    const ion7_csampler_params_t* p,
    const char*                   grammar,
    const char**                  trigger_words,
    int                           n_triggers,
    const int32_t*                logit_bias_ids,
    const float*                  logit_bias_val,
    int                           n_logit_bias)
{
    common_params_sampling sp;

    if (p) {
        sp.seed            = p->seed;
        sp.top_k           = p->top_k;
        sp.top_p           = p->top_p;
        sp.min_p           = p->min_p;
        sp.xtc_probability = p->xtc_probability;
        sp.xtc_threshold   = p->xtc_threshold;
        sp.temp            = p->temp;
        sp.penalty_repeat  = p->repeat_penalty;
        sp.penalty_freq    = p->freq_penalty;
        sp.penalty_present = p->pres_penalty;
        sp.penalty_last_n  = p->repeat_last_n;
        sp.dry_multiplier      = p->dry_mult;
        sp.dry_base            = p->dry_base;
        sp.dry_allowed_length  = p->dry_allowed_len;
        sp.dry_penalty_last_n  = p->dry_last_n;
        sp.mirostat        = p->mirostat;
        sp.mirostat_tau    = p->mirostat_tau;
        sp.mirostat_eta    = p->mirostat_eta;
        sp.grammar_lazy    = (bool)p->grammar_lazy;
    }

    if (grammar && grammar[0] != '\0')
        sp.grammar = common_grammar(COMMON_GRAMMAR_TYPE_USER, grammar);

    for (int i = 0; i < n_triggers && trigger_words; i++) {
        if (trigger_words[i]) {
            common_grammar_trigger trig;
            trig.type  = COMMON_GRAMMAR_TRIGGER_TYPE_WORD;
            trig.value = trigger_words[i];
            sp.grammar_triggers.push_back(std::move(trig));
        }
    }

    for (int i = 0; i < n_logit_bias && logit_bias_ids && logit_bias_val; i++) {
        llama_logit_bias lb;
        lb.token = logit_bias_ids[i];
        lb.bias  = logit_bias_val[i];
        sp.logit_bias.push_back(lb);
    }

    auto* w = new ion7_csampler;
    w->smpl = common_sampler_init(model, sp);
    if (!w->smpl) { delete w; return nullptr; }
    return w;
}

void ion7_csampler_free(ion7_csampler_t* s)
{
    if (!s) return;
    common_sampler_free(s->smpl);
    delete s;
}

int32_t ion7_csampler_sample(ion7_csampler_t* s, struct llama_context* ctx,
                              int idx, int grammar_first)
{
    if (!s) return -1;
    return common_sampler_sample(s->smpl, ctx, idx, (bool)grammar_first);
}

void ion7_csampler_accept(ion7_csampler_t* s, int32_t token)
{
    if (s) common_sampler_accept(s->smpl, token, true);
}

void ion7_csampler_reset(ion7_csampler_t* s)
{
    if (s) common_sampler_reset(s->smpl);
}

int32_t ion7_csampler_last(const ion7_csampler_t* s)
{
    if (!s) return -1;
    return common_sampler_last(s->smpl);
}

uint32_t ion7_csampler_get_seed(const ion7_csampler_t* s)
{
    if (!s) return 0;
    return common_sampler_get_seed(s->smpl);
}

/* =========================================================================
 * ── Speculative decoding ─────────────────────────────────────────────────
 * ======================================================================= */

struct ion7_speculative {
    common_speculative*      spec;
    common_params_speculative params;
};

ion7_speculative_t* ion7_speculative_init(
    struct llama_context* ctx_tgt,
    struct llama_context* ctx_dft,
    int type, int n_draft, int ngram_min, int ngram_max)
{
    auto* w = new ion7_speculative;
    w->params.type      = (common_speculative_type)type;
    w->params.n_max     = (n_draft > 0) ? n_draft : 16;
    w->params.n_min     = 0;
    /* ngram_min/ngram_max map to the n-gram table sizes */
    if (ngram_min > 0) w->params.ngram_size_n = (uint16_t)ngram_min;
    if (ngram_max > 0) w->params.ngram_size_m = (uint16_t)ngram_max;
    /* For ION7_SPEC_DRAFT the draft model is taken from the draft context */
    if (ctx_dft)
        w->params.model_dft = const_cast<llama_model*>(llama_get_model(ctx_dft));
    w->spec = common_speculative_init(w->params, ctx_tgt);
    if (!w->spec) { delete w; return nullptr; }
    return w;
}

void ion7_speculative_free(ion7_speculative_t* spec)
{
    if (!spec) return;
    common_speculative_free(spec->spec);
    delete spec;
}

void ion7_speculative_begin(ion7_speculative_t* spec,
                             const int32_t* prompt, int n_prompt)
{
    if (!spec || !prompt) return;
    llama_tokens toks(prompt, prompt + n_prompt);
    common_speculative_begin(spec->spec, toks);
}

int ion7_speculative_draft(ion7_speculative_t* spec,
                            const int32_t* prompt, int n_prompt,
                            int32_t last_token,
                            int32_t* out_draft, int max_draft)
{
    if (!spec || !prompt || !out_draft) return 0;
    llama_tokens toks(prompt, prompt + n_prompt);
    llama_tokens draft = common_speculative_draft(
        spec->spec, spec->params, toks, last_token);
    int n = (int)std::min((size_t)max_draft, draft.size());
    for (int i = 0; i < n; i++) out_draft[i] = draft[i];
    return n;
}

void ion7_speculative_accept(ion7_speculative_t* spec, int n_accepted)
{
    if (spec) common_speculative_accept(spec->spec, (uint16_t)n_accepted);
}

void ion7_speculative_stats(const ion7_speculative_t* spec)
{
    if (spec) common_speculative_print_stats(spec->spec);
}

/* =========================================================================
 * ── Chat output parsing ───────────────────────────────────────────────────
 * ======================================================================= */

static void safe_copy(const std::string& src, char* dst, int32_t cap, int* truncated)
{
    if (!dst || cap <= 0) return;
    size_t n = src.size();
    if ((int32_t)n >= cap) { n = (size_t)(cap - 1); *truncated = 1; }
    memcpy(dst, src.c_str(), n);
    dst[n] = '\0';
}

int ion7_chat_parse(
    ion7_chat_templates_t* t,
    const char*            text,
    int                    enable_thinking,
    char*   content_buf,  int32_t content_len,
    char*   thinking_buf, int32_t thinking_len,
    char*   tools_buf,    int32_t tools_len,
    int*    out_has_tools)
{
    if (!t || !text) return -1;

    common_chat_parser_params pparams;
    pparams.parse_tool_calls  = true;
    pparams.reasoning_format  = (enable_thinking != 0)
                                ? COMMON_REASONING_FORMAT_AUTO
                                : COMMON_REASONING_FORMAT_NONE;

    common_chat_msg msg = common_chat_parse(text, false, pparams);

    int truncated = 0;
    safe_copy(msg.content,            content_buf,  content_len,  &truncated);
    safe_copy(msg.reasoning_content,  thinking_buf, thinking_len, &truncated);

    /* Serialise tool_calls to a JSON array string */
    std::ostringstream tools_json;
    tools_json << "[";
    for (size_t i = 0; i < msg.tool_calls.size(); i++) {
        if (i > 0) tools_json << ",";
        const auto& tc = msg.tool_calls[i];
        tools_json << "{\"name\":\"" << tc.name
                   << "\",\"arguments\":" << tc.arguments
                   << ",\"id\":\"" << tc.id << "\"}";
    }
    tools_json << "]";
    safe_copy(tools_json.str(), tools_buf, tools_len, &truncated);

    if (out_has_tools) *out_has_tools = msg.tool_calls.empty() ? 0 : 1;
    return truncated ? 1 : 0;
}

/* =========================================================================
 * ── UTF-8 helpers ────────────────────────────────────────────────────────
 * ======================================================================= */

int ion7_utf8_seq_len(uint8_t b)
{
    if ((b & 0x80) == 0x00) return 1;
    if ((b & 0xE0) == 0xC0) return 2;
    if ((b & 0xF0) == 0xE0) return 3;
    if ((b & 0xF8) == 0xF0) return 4;
    return 0;  /* continuation byte or invalid */
}

int ion7_utf8_is_complete(const char* buf, size_t len)
{
    if (!buf || len == 0) return 1;
    const uint8_t* p = (const uint8_t*)buf;
    size_t i = 0;
    while (i < len) {
        int seq = ion7_utf8_seq_len(p[i]);
        if (seq == 0) return 0;  /* invalid */
        if (i + (size_t)seq > len) return 0;  /* incomplete */
        i += seq;
    }
    return 1;
}

/* =========================================================================
 * ── JSON Schema → GBNF ───────────────────────────────────────────────────
 * ======================================================================= */

int ion7_json_schema_to_grammar(const char* schema_json, char* out, size_t out_len)
{
    if (!schema_json) return -1;
    try {
        auto schema = nlohmann::ordered_json::parse(schema_json);
        std::string gbnf = json_schema_to_grammar(schema);
        size_t needed = gbnf.size() + 1;
        if (out && out_len >= needed)
            memcpy(out, gbnf.c_str(), needed);
        return (int)needed;
    } catch (...) {
        return -1;
    }
}

/* =========================================================================
 * ── Partial regex ────────────────────────────────────────────────────────
 * ======================================================================= */

struct ion7_regex {
    common_regex rx;
    explicit ion7_regex(const char* p) : rx(p) {}
};

ion7_regex_t* ion7_regex_new(const char* pattern)
{
    if (!pattern) return nullptr;
    try { return new ion7_regex(pattern); }
    catch (...) { return nullptr; }
}

void ion7_regex_free(ion7_regex_t* r) { delete r; }

int ion7_regex_search(ion7_regex_t* r, const char* text, size_t len, int partial)
{
    if (!r || !text) return 0;
    std::string s(text, len);
    common_regex_match m = r->rx.search(s, 0, /*as_match=*/false);
    if (m.type == COMMON_REGEX_MATCH_TYPE_FULL)    return 2;
    if (partial && m.type == COMMON_REGEX_MATCH_TYPE_PARTIAL) return 1;
    return 0;
}

/* =========================================================================
 * ── Control vectors ──────────────────────────────────────────────────────
 * ======================================================================= */

int ion7_cvec_apply(struct llama_context* ctx,
                     const float* data, size_t len,
                     int32_t n_embd, int32_t il_start, int32_t il_end)
{
    if (!ctx || !data) return -1;
    return (int)llama_set_adapter_cvec(ctx, data, len, n_embd, il_start, il_end);
}

void ion7_cvec_clear(struct llama_context* ctx)
{
    if (ctx) llama_set_adapter_cvec(ctx, nullptr, 0, 0, -1, -1);
}

/* =========================================================================
 * ── NUMA ─────────────────────────────────────────────────────────────────
 * ======================================================================= */

void ion7_numa_init(int strategy)
{
    ggml_numa_init((enum ggml_numa_strategy)strategy);
}

int ion7_is_numa(void)
{
    return ggml_is_numa() ? 1 : 0;
}

/* =========================================================================
 * ── CPU capability detection ─────────────────────────────────────────────
 * ======================================================================= */

void ion7_cpu_caps(ion7_cpu_caps_t* out)
{
    if (!out) return;
    memset(out, 0, sizeof(*out));
    out->sse3         = ggml_cpu_has_sse3();
    out->ssse3        = ggml_cpu_has_ssse3();
    out->avx          = ggml_cpu_has_avx();
    out->avx2         = ggml_cpu_has_avx2();
    out->avx_vnni     = ggml_cpu_has_avx_vnni();
    out->bmi2         = ggml_cpu_has_bmi2();
    out->f16c         = ggml_cpu_has_f16c();
    out->fma          = ggml_cpu_has_fma();
    out->avx512       = ggml_cpu_has_avx512();
    out->avx512_vbmi  = ggml_cpu_has_avx512_vbmi();
    out->avx512_vnni  = ggml_cpu_has_avx512_vnni();
    out->avx512_bf16  = ggml_cpu_has_avx512_bf16();
    out->amx_int8     = ggml_cpu_has_amx_int8();
    out->neon         = ggml_cpu_has_neon();
    out->arm_fma      = ggml_cpu_has_arm_fma();
    out->fp16_va      = ggml_cpu_has_fp16_va();
    out->dotprod      = ggml_cpu_has_dotprod();
    out->matmul_int8  = ggml_cpu_has_matmul_int8();
    out->sve          = ggml_cpu_has_sve();
    out->sve_cnt      = ggml_cpu_get_sve_cnt();
    out->sme          = ggml_cpu_has_sme();
    out->riscv_v      = ggml_cpu_has_riscv_v();
    out->rvv_vlen     = ggml_cpu_get_rvv_vlen();
    out->vsx          = ggml_cpu_has_vsx();
    out->wasm_simd    = ggml_cpu_has_wasm_simd();
}

/* =========================================================================
 * ── Log routing ──────────────────────────────────────────────────────────
 * ======================================================================= */

static FILE*  g_log_file       = nullptr;
static int    g_log_timestamps = 0;

static void ion7_log_cb_full(enum ggml_log_level level, const char* text, void* ud)
{
    (void)ud;
    if (g_log_level <= 0) return;
    int threshold = 5 - g_log_level;
    if ((int)level < threshold) return;

    FILE* dst = g_log_file ? g_log_file : stderr;
    if (g_log_timestamps) {
        auto now = std::chrono::system_clock::now();
        auto t   = std::chrono::system_clock::to_time_t(now);
        struct tm tm_buf;
#ifdef _WIN32
        localtime_s(&tm_buf, &t);
#else
        localtime_r(&t, &tm_buf);
#endif
        char ts[32];
        strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S ", &tm_buf);
        fputs(ts, dst);
    }
    fputs(text, dst);
    if (g_log_file) fflush(g_log_file);
}

void ion7_log_to_file(const char* path)
{
    if (g_log_file && g_log_file != stderr) {
        fclose(g_log_file);
        g_log_file = nullptr;
    }
    if (path && path[0] != '\0') {
        g_log_file = fopen(path, "a");
    }
    /* Re-register the callback so it picks up the new file */
    llama_log_set(ion7_log_cb_full, nullptr);
}

void ion7_log_set_timestamps(int enable)
{
    g_log_timestamps = enable ? 1 : 0;
    llama_log_set(ion7_log_cb_full, nullptr);
}

/* =========================================================================
 * ── Base64 ───────────────────────────────────────────────────────────────
 * ======================================================================= */

int ion7_base64_encode(const uint8_t* data, size_t len, char* out, size_t out_len)
{
    if (!data || !out) return -1;
    /* Required output: ceil(len/3)*4 + 1 */
    size_t needed = ((len + 2) / 3) * 4 + 1;
    if (out_len < needed) return -1;
    std::string enc = base64::encode(reinterpret_cast<const char*>(data), len);
    memcpy(out, enc.c_str(), enc.size() + 1);
    return (int)enc.size();
}

int ion7_base64_decode(const char* src, size_t src_len, uint8_t* out, size_t out_len)
{
    if (!src || !out) return -1;
    try {
        std::string dec = base64::decode(src, src_len);
        if (out_len < dec.size()) return -1;
        memcpy(out, dec.data(), dec.size());
        return (int)dec.size();
    } catch (...) {
        return -1;
    }
}

/* =========================================================================
 * ── JSON utilities ────────────────────────────────────────────────────────
 * ======================================================================= */

int ion7_json_validate(const char* json_str)
{
    if (!json_str) return 0;
    return nlohmann::ordered_json::accept(json_str) ? 1 : 0;
}

int ion7_json_format(const char* json_str, char* out, size_t out_len)
{
    if (!json_str) return -1;
    try {
        auto j = nlohmann::ordered_json::parse(json_str);
        std::string s = j.dump(2);
        size_t needed = s.size() + 1;
        if (out && out_len >= needed)
            memcpy(out, s.c_str(), needed);
        return (int)needed;
    } catch (...) { return -1; }
}

int ion7_json_merge(const char* base, const char* overlay,
                     char* out, size_t out_len)
{
    if (!base || !overlay) return -1;
    try {
        auto j = nlohmann::ordered_json::parse(base);
        j.merge_patch(nlohmann::ordered_json::parse(overlay));
        std::string s = j.dump();
        size_t needed = s.size() + 1;
        if (out && out_len >= needed)
            memcpy(out, s.c_str(), needed);
        return (int)needed;
    } catch (...) { return -1; }
}
