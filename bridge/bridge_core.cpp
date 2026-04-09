/* bridge_core.cpp - llama.h layer (model, context, KV, state, samplers) */
/*
 * Copyright (C) 2026 Ion7 Project Contributors
 * SPDX-License-Identifier: MIT
 *
 * This file is part of ion7-core.
 *
 * ──────────────────────────────────────────────────────────────────────────
 * Migrated from C to C++ (ion7-core v1.1) to enable libcommon integration.
 * The public API is unchanged — all symbols remain extern "C".
 * ──────────────────────────────────────────────────────────────────────────
 */

#include "ion7_bridge.h"
#include "llama.h"
#include "ggml-cpu.h"
#include "bridge_internal.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <string>

/* =========================================================================
 * Log globals (definitions - declared extern in bridge_internal.hpp)
 * ======================================================================= */

int   g_log_level      = 1;
FILE* g_log_file       = nullptr;
int   g_log_timestamps = 0;

/* =========================================================================
 * Unified log callback
 * ======================================================================= */

void ion7_log_dispatch(enum ggml_log_level level, const char* text, void* ud)
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

/* =========================================================================
 * Version & info
 * ======================================================================= */

const char* ion7_bridge_version(void) { return "1.1.0"; }
const char* ion7_llama_info(void)     { return llama_print_system_info(); }

/* =========================================================================
 * Backend
 * ======================================================================= */

void ion7_backend_init(void) { llama_log_set(ion7_log_dispatch, NULL); llama_backend_init(); }
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

struct llama_model* ion7_model_load(const char* path, int32_t n_gpu_layers, int use_mmap, int use_mlock, int vocab_only)
{
    struct llama_model_params p = llama_model_default_params();
    p.n_gpu_layers = n_gpu_layers;
    p.use_mmap     = (bool)use_mmap;
    p.use_mlock    = (bool)use_mlock;
    p.vocab_only   = (bool)vocab_only;
    return llama_model_load_from_file(path, p);
}

void ion7_model_free(struct llama_model* m) { if (m) llama_model_free(m); }

struct llama_model* ion7_model_load_splits(const char** paths, size_t n_paths, int32_t n_gpu_layers)
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

int ion7_model_meta_val(const struct llama_model* m, const char* key, char* buf, size_t sz)
{
    return llama_model_meta_val_str(m, key, buf, sz);
}

int ion7_model_meta_count(const struct llama_model* m)
{
    return llama_model_meta_count(m);
}

int ion7_model_meta_key_at(const struct llama_model* m, int32_t idx, char* buf, size_t sz)
{
    return llama_model_meta_key_by_index(m, idx, buf, sz);
}

int ion7_model_meta_val_at(const struct llama_model* m, int32_t idx, char* buf, size_t sz)
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
uint32_t ion7_model_n_cls_out(const struct llama_model* m) { return llama_model_n_cls_out(m); }
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

int ion7_params_fit(const char* path, int32_t* n_gpu_layers, uint32_t* n_ctx, uint32_t n_ctx_min)
{
    struct llama_model_params   mparams = llama_model_default_params();
    struct llama_context_params cparams = llama_context_default_params();

    cparams.n_ctx = *n_ctx;

    size_t max_dev = llama_max_devices();
    size_t max_ovr = llama_max_tensor_buft_overrides();
    float* ts      = (float*)calloc(max_dev, sizeof(float));
    struct llama_model_tensor_buft_override* ov = (struct llama_model_tensor_buft_override*)calloc(max_ovr, sizeof(struct llama_model_tensor_buft_override));
    size_t* margins = (size_t*)calloc(max_dev, sizeof(size_t));

    enum llama_params_fit_status status = llama_params_fit(path, &mparams, &cparams, ts, ov, margins, n_ctx_min, GGML_LOG_LEVEL_WARN);

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

/* llama_context_params builder shared by ion7_context_create variants */
static struct llama_context_params build_ctx_params(
    uint32_t n_ctx, uint32_t n_batch, uint32_t n_ubatch, uint32_t n_seq_max,
    int32_t n_threads, int32_t n_threads_batch,
    int flash_attn, int offload_kqv, int op_offload,
    int no_perf, int type_k, int type_v, int swa_full, int kv_unified)
{
    struct llama_context_params p = llama_context_default_params();
    p.n_ctx           = n_ctx     ? n_ctx     : 4096;
    p.n_batch         = n_batch   ? n_batch   : 2048;
    p.n_ubatch        = n_ubatch  ? n_ubatch  : p.n_batch;
    p.n_seq_max       = n_seq_max ? n_seq_max : 1;
    p.n_threads       = n_threads       > 0 ? n_threads       : 4;
    p.n_threads_batch = n_threads_batch > 0 ? n_threads_batch : p.n_threads * 2;
    /* quantised KV (type_k != 1 = not F16) requires explicit flash attention */
    p.flash_attn_type = (flash_attn || type_k != 1)
                            ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                            : LLAMA_FLASH_ATTN_TYPE_AUTO;
    p.offload_kqv = (bool)offload_kqv;
    p.op_offload  = (bool)op_offload;
    p.no_perf     = (bool)no_perf;
    p.swa_full    = (bool)swa_full;
    p.kv_unified  = (bool)kv_unified;
    if (type_k > 0) p.type_k = (enum ggml_type)type_k;
    if (type_v > 0) p.type_v = (enum ggml_type)type_v;
    return p;
}

struct llama_context* ion7_context_create(
    struct llama_model* model,
    uint32_t n_ctx, uint32_t n_batch, uint32_t n_ubatch, uint32_t n_seq_max,
    int32_t n_threads, int32_t n_threads_batch,
    int flash_attn, int offload_kqv, int op_offload,
    int no_perf, int type_k, int type_v, int swa_full, int kv_unified)
{
    struct llama_context_params p = build_ctx_params(
        n_ctx, n_batch, n_ubatch, n_seq_max,
        n_threads, n_threads_batch,
        flash_attn, offload_kqv, op_offload,
        no_perf, type_k, type_v, swa_full, kv_unified);
    return llama_init_from_model(model, p);
}

struct llama_context* ion7_context_create_with_cb(
    struct llama_model* model,
    uint32_t n_ctx, uint32_t n_batch, uint32_t n_ubatch, uint32_t n_seq_max,
    int32_t n_threads, int32_t n_threads_batch,
    int flash_attn, int offload_kqv, int op_offload,
    int no_perf, int type_k, int type_v, int swa_full, int kv_unified,
    void* cb_eval, void* cb_eval_user_data)
{
    struct llama_context_params p = build_ctx_params(
        n_ctx, n_batch, n_ubatch, n_seq_max,
        n_threads, n_threads_batch,
        flash_attn, offload_kqv, op_offload,
        no_perf, type_k, type_v, swa_full, kv_unified);
    p.cb_eval           = (ggml_backend_sched_eval_callback)cb_eval;
    p.cb_eval_user_data = cb_eval_user_data;
    return llama_init_from_model(model, p);
}

struct llama_context* ion7_embedding_context_create(struct llama_model* model, uint32_t n_ctx, uint32_t n_seq_max, int32_t n_threads, int pooling)
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
void ion7_kv_seq_cp(struct llama_context* ctx, int32_t src, int32_t dst, int32_t p0, int32_t p1)
{
    llama_memory_t m = ctx_mem(ctx);
    if (m) llama_memory_seq_cp(m, src, dst, p0, p1);
}
void ion7_kv_seq_keep(struct llama_context* ctx, int32_t seq)
{
    llama_memory_t m = ctx_mem(ctx);
    if (m) llama_memory_seq_keep(m, seq);
}
void ion7_kv_seq_shift(struct llama_context* ctx, int32_t seq, int32_t p0, int32_t p1, int32_t delta)
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
int ion7_state_save_file(struct llama_context* ctx, const char* path, const llama_token* tokens, size_t n)
{
    return llama_state_save_file(ctx, path, tokens, n) ? 1 : 0;
}
int ion7_state_load_file(struct llama_context* ctx, const char* path, llama_token* out, size_t cap, size_t* n_out)
{
    return llama_state_load_file(ctx, path, out, cap, n_out) ? 1 : 0;
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
int ion7_lora_meta_val(const struct llama_adapter_lora* a, const char* key, char* buf, size_t sz)
{
    return llama_adapter_meta_val_str(a, key, buf, sz);
}

/* =========================================================================
 * Performance
 * ======================================================================= */

void ion7_perf_print(struct llama_context* ctx) { llama_perf_context_print(ctx); }
void ion7_perf_reset(struct llama_context* ctx) { llama_perf_context_reset(ctx); }

void ion7_perf_get(struct llama_context* ctx, double* t_load_ms, double* t_p_eval_ms, double*  t_eval_ms, int32_t* n_p_eval, int32_t* n_eval, int32_t* n_reused)
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

void ion7_threadpool_attach(struct llama_context* ctx, ion7_threadpool_t* tp, ion7_threadpool_t* tp_batch)
{
    llama_attach_threadpool(ctx, (ggml_threadpool_t)tp, tp_batch ? (ggml_threadpool_t)tp_batch : (ggml_threadpool_t)tp);
}

void ion7_threadpool_detach(struct llama_context* ctx)
{
    llama_detach_threadpool(ctx);
}

int ion7_threadpool_n_threads(ion7_threadpool_t* tp)
{
    if (!tp) return 0;
    return ggml_threadpool_get_n_threads((ggml_threadpool_t)tp);
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

static const char* ion7_smpl_name(const struct llama_sampler* smpl) {
    return ((ion7_custom_sampler_ctx_t*)smpl->ctx)->name;
}
static void ion7_smpl_accept(struct llama_sampler* smpl, llama_token tok) {
    ion7_custom_sampler_ctx_t* c = (ion7_custom_sampler_ctx_t*)smpl->ctx;
    if (c->accept_fn) c->accept_fn(tok, c->userdata);
}
static void ion7_smpl_apply(struct llama_sampler* smpl, llama_token_data_array* cur_p) {
    ion7_custom_sampler_ctx_t* c = (ion7_custom_sampler_ctx_t*)smpl->ctx;
    if (c->apply_fn) c->apply_fn(cur_p, c->userdata);
}
static void ion7_smpl_reset(struct llama_sampler* smpl) {
    ion7_custom_sampler_ctx_t* c = (ion7_custom_sampler_ctx_t*)smpl->ctx;
    if (c->reset_fn) c->reset_fn(c->userdata);
}
static void ion7_smpl_free(struct llama_sampler* smpl) {
    ion7_custom_sampler_ctx_t* c = (ion7_custom_sampler_ctx_t*)smpl->ctx;
    if (c->free_fn) c->free_fn(c->userdata);
    free(c);
}
static struct llama_sampler* ion7_smpl_clone(const struct llama_sampler* smpl) {
    ion7_custom_sampler_ctx_t* src = (ion7_custom_sampler_ctx_t*)smpl->ctx;
    return ion7_sampler_create(
        src->name, src->apply_fn, src->accept_fn,
        src->reset_fn, src->free_fn, src->userdata);
}

/* Field order matches struct llama_sampler_i exactly:
 * name, accept, apply, reset, clone, free, backend_init,
 * backend_accept, backend_apply, backend_set_input          */
static struct llama_sampler_i ion7_custom_iface = {
    ion7_smpl_name,    /* name   */
    ion7_smpl_accept,  /* accept */
    ion7_smpl_apply,   /* apply  */
    ion7_smpl_reset,   /* reset  */
    ion7_smpl_clone,   /* clone  */
    ion7_smpl_free,    /* free   */
    nullptr,           /* backend_init      */
    nullptr,           /* backend_accept    */
    nullptr,           /* backend_apply     */
    nullptr,           /* backend_set_input */
};

struct llama_sampler* ion7_sampler_create(
    const char*            name,
    ion7_sampler_apply_fn  apply_fn,
    ion7_sampler_accept_fn accept_fn,
    ion7_sampler_reset_fn  reset_fn,
    ion7_sampler_free_fn   free_fn,
    void*                  userdata)
{
    ion7_custom_sampler_ctx_t* ctx = (ion7_custom_sampler_ctx_t*)malloc(sizeof(ion7_custom_sampler_ctx_t));
    if (!ctx) return nullptr;
    strncpy(ctx->name, name ? name : "custom", sizeof(ctx->name) - 1);
    ctx->name[sizeof(ctx->name) - 1] = '\0';
    ctx->apply_fn  = apply_fn;
    ctx->accept_fn = accept_fn;
    ctx->reset_fn  = reset_fn;
    ctx->free_fn   = free_fn;
    ctx->userdata  = userdata;
    return llama_sampler_init(&ion7_custom_iface, (llama_sampler_context_t)ctx);
}

/* =========================================================================
 * Model quantization
 * ======================================================================= */

int ion7_model_quantize(const char* path_in, const char* path_out, int ftype, int n_threads, int pure, int allow_req, int dry_run)
{
    llama_model_quantize_params p = llama_model_quantize_default_params();
    p.ftype            = (enum llama_ftype)ftype;
    p.nthread          = n_threads;
    p.pure             = (bool)pure;
    p.allow_requantize = (bool)allow_req;
    p.dry_run          = (bool)dry_run;
    return (int)llama_model_quantize(path_in, path_out, &p);
}
