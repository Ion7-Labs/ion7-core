--- @module ion7.core.ffi.loader
--- SPDX-License-Identifier: MIT
---
--- Loads libllama.so and ion7_bridge.so, initialises the backend.
---
--- Path resolution order for each library:
---   1. Explicit path in opts (opts.llama_lib / opts.bridge_lib)
---   2. Environment variable LLAMA_LIB / ION7_BRIDGE
---   3. Default: system paths (/usr/local/lib, /usr/lib) - no hardcoded fallback
---
--- @usage
---   local Loader = require "ion7.core.ffi.loader"
---   local L = Loader.init({ log_level = 0 })
---   -- L.lib    -> libllama FFI namespace
---   -- L.bridge -> ion7_bridge FFI namespace
---   -- L.ffi    -> ffi module
---   -- L.C      -> constants

local types = require "ion7.core.ffi.types"
local ffi   = types.ffi

-- ── Bridge cdef ───────────────────────────────────────────────────────────────

ffi.cdef [[
/* ---- Version & backend ------------------------------------------------- */
const char*   ion7_bridge_version(void);
const char*   ion7_llama_info    (void);
void          ion7_backend_init  (void);
void          ion7_backend_free  (void);
void          ion7_set_log_level (int level);

/* ---- Capabilities ------------------------------------------------------- */
int           ion7_supports_mmap          (void);
int           ion7_supports_mlock         (void);
int           ion7_supports_gpu_offload   (void);
int           ion7_supports_rpc           (void);
int           ion7_max_devices            (void);
int           ion7_max_parallel_sequences (void);

/* ---- Model -------------------------------------------------------------- */
struct llama_model* ion7_model_load(const char* path, int32_t n_gpu_layers,
                                    int use_mmap, int use_mlock, int vocab_only);
struct llama_model* ion7_model_load_fd    (int fd, int n_gpu_layers);
struct llama_model* ion7_model_load_splits(const char** paths, size_t n_paths,
                                           int32_t n_gpu_layers);
void          ion7_model_free    (struct llama_model* model);
void          ion7_model_save    (const struct llama_model* m, const char* path);
int           ion7_model_desc    (const struct llama_model* m, char* buf, size_t sz);
uint64_t      ion7_model_size    (const struct llama_model* m);
uint64_t      ion7_model_n_params(const struct llama_model* m);
int           ion7_model_meta_val   (const struct llama_model* m, const char* key, char* buf, size_t sz);
int           ion7_model_meta_count (const struct llama_model* m);
int           ion7_model_meta_key_at(const struct llama_model* m, int32_t idx, char* buf, size_t sz);
int           ion7_model_meta_val_at(const struct llama_model* m, int32_t idx, char* buf, size_t sz);
const char*   ion7_model_chat_template(const struct llama_model* m, const char* name);
int           ion7_model_has_encoder   (const struct llama_model* m);
int           ion7_model_has_decoder   (const struct llama_model* m);
int           ion7_model_is_recurrent  (const struct llama_model* m);
int           ion7_model_is_hybrid     (const struct llama_model* m);
int           ion7_model_is_diffusion  (const struct llama_model* m);
int32_t       ion7_model_n_ctx_train   (const struct llama_model* m);
int32_t       ion7_model_n_embd        (const struct llama_model* m);
int32_t       ion7_model_n_embd_inp    (const struct llama_model* m);
int32_t       ion7_model_n_embd_out    (const struct llama_model* m);
int32_t       ion7_model_n_layer       (const struct llama_model* m);
int32_t       ion7_model_n_head        (const struct llama_model* m);
int32_t       ion7_model_n_head_kv     (const struct llama_model* m);
int32_t       ion7_model_n_swa         (const struct llama_model* m);
uint32_t      ion7_model_n_cls_out     (const struct llama_model* m);
float         ion7_model_rope_freq_scale_train(const struct llama_model* m);
const char*   ion7_model_rope_type     (const struct llama_model* m);
const char*   ion7_model_cls_label     (const struct llama_model* m, uint32_t i);
int32_t       ion7_model_decoder_start_token(const struct llama_model* m);
int           ion7_params_fit          (const char* path, int32_t* n_gpu_layers, uint32_t* n_ctx, uint32_t n_ctx_min);
int           ion7_model_quantize      (const char* path_in, const char* path_out, int ftype, int n_threads, int pure, int allow_req, int quant_out, int dry_run);

/* ---- Context ------------------------------------------------------------ */
struct llama_context* ion7_context_create(
    struct llama_model* model,
    uint32_t n_ctx, uint32_t n_batch, uint32_t n_ubatch,
    uint32_t n_seq_max,
    int32_t n_threads, int32_t n_threads_batch,
    int flash_attn, int offload_kqv, int op_offload,
    int no_perf, int type_k, int type_v, int swa_full, int kv_unified);
struct llama_context* ion7_embedding_context_create(
    struct llama_model* model, uint32_t n_ctx, uint32_t n_seq_max,
    int32_t n_threads, int pooling);
struct llama_context* ion7_context_create_with_cb(
    struct llama_model* model,
    uint32_t n_ctx, uint32_t n_batch, uint32_t n_ubatch,
    uint32_t n_seq_max,
    int32_t n_threads, int32_t n_threads_batch,
    int flash_attn, int offload_kqv, int op_offload,
    int no_perf, int type_k, int type_v, int swa_full, int kv_unified,
    void* cb_eval, void* cb_eval_user_data);
void     ion7_context_free(struct llama_context* ctx);

/* ---- KV cache ----------------------------------------------------------- */
void ion7_kv_clear    (struct llama_context* ctx);
void ion7_kv_seq_rm   (struct llama_context* ctx, int32_t seq, int32_t p0, int32_t p1);
void ion7_kv_seq_cp   (struct llama_context* ctx, int32_t src, int32_t dst,
                        int32_t p0, int32_t p1);
void ion7_kv_seq_keep (struct llama_context* ctx, int32_t seq);
void ion7_kv_seq_shift(struct llama_context* ctx, int32_t seq,
                        int32_t p0, int32_t p1, int32_t delta);

/* ---- State -------------------------------------------------------------- */
size_t ion7_state_size     (struct llama_context* ctx);
size_t ion7_state_get      (struct llama_context* ctx, uint8_t* buf, size_t sz);
int    ion7_state_set      (struct llama_context* ctx, const uint8_t* buf, size_t sz);
int    ion7_state_save_file(struct llama_context* ctx, const char* path,
                             const llama_token* tokens, size_t n);
int    ion7_state_load_file(struct llama_context* ctx, const char* path,
                             llama_token* out, size_t cap, size_t* n_out);
size_t ion7_state_seq_size (struct llama_context* ctx, int32_t seq_id);
int    ion7_state_seq_save (struct llama_context* ctx, const char* path, int32_t seq_id);
int    ion7_state_seq_load (struct llama_context* ctx, const char* path, int32_t dest_seq_id);

/* ---- LoRA --------------------------------------------------------------- */
struct llama_adapter_lora* ion7_lora_load    (struct llama_model* model, const char* path);
void   ion7_lora_free    (struct llama_adapter_lora* adapter);
int    ion7_lora_apply   (struct llama_context* ctx, struct llama_adapter_lora* a, float scale);
int    ion7_lora_remove  (struct llama_context* ctx);
int    ion7_lora_meta_val(const struct llama_adapter_lora* a, const char* key,
                           char* buf, size_t sz);

/* ---- Chat templates (Jinja2, libcommon) --------------------------------- */
typedef struct common_chat_templates ion7_chat_templates_t;
ion7_chat_templates_t* ion7_chat_templates_init   (const struct llama_model* model, const char* tmpl_override);
void    ion7_chat_templates_free   (ion7_chat_templates_t* t);
int     ion7_chat_templates_support_thinking(const ion7_chat_templates_t* t);
int32_t ion7_chat_templates_apply  (ion7_chat_templates_t* t,
                                    const char** roles,
                                    const char** contents,
                                    size_t n_msgs,
                                    int add_ass,
                                    int enable_thinking,
                                    char* buf, int32_t buf_len);

/* ---- Reasoning budget --------------------------------------------------- */
struct llama_sampler* ion7_reasoning_budget_init(const struct llama_model* model, int32_t n_budget);

/* ---- Training (llama_opt) ----------------------------------------------- */
typedef struct ion7_opt_state ion7_opt_state_t;
typedef struct ggml_opt_dataset* ggml_opt_dataset_t;
ion7_opt_state_t*  ion7_opt_init (struct llama_context* ctx, struct llama_model* model, int optimizer, float lr);
void ion7_opt_free (ion7_opt_state_t* state);
ggml_opt_dataset_t ion7_opt_dataset_create(struct llama_context* ctx, const llama_token* tokens, int64_t n_tokens, int64_t n_ctx);
void ion7_opt_dataset_free (ggml_opt_dataset_t dataset);
float ion7_opt_epoch (struct llama_context* ctx, ggml_opt_dataset_t dataset, float val_split);

/* ---- Threadpool --------------------------------------------------------- */
typedef struct ggml_threadpool ion7_threadpool_t;
ion7_threadpool_t* ion7_threadpool_create (int n_threads);
void               ion7_threadpool_free   (ion7_threadpool_t* tp);
void               ion7_threadpool_pause  (ion7_threadpool_t* tp);
void               ion7_threadpool_resume (ion7_threadpool_t* tp);
void               ion7_threadpool_attach (struct llama_context* ctx,
                                            ion7_threadpool_t* tp,
                                            ion7_threadpool_t* tp_batch);
void               ion7_threadpool_detach (struct llama_context* ctx);
int                ion7_threadpool_n_threads(ion7_threadpool_t* tp);

/* ---- Custom sampler ----------------------------------------------------- */
typedef void (*ion7_sampler_apply_fn) (llama_token_data_array* cur_p, void* ud);
typedef void (*ion7_sampler_accept_fn)(llama_token token,              void* ud);
typedef void (*ion7_sampler_reset_fn) (void*                           ud);
typedef void (*ion7_sampler_free_fn)  (void*                           ud);
struct llama_sampler* ion7_sampler_create(
    const char*            name,
    ion7_sampler_apply_fn  apply_fn,
    ion7_sampler_accept_fn accept_fn,
    ion7_sampler_reset_fn  reset_fn,
    ion7_sampler_free_fn   free_fn,
    void*                  userdata);

/* ---- Perf --------------------------------------------------------------- */
void ion7_perf_print(struct llama_context* ctx);
void ion7_perf_reset(struct llama_context* ctx);
void ion7_perf_get  (struct llama_context* ctx,
                      double*  t_load_ms,   double*  t_p_eval_ms,
                      double*  t_eval_ms,   int32_t* n_p_eval,
                      int32_t* n_eval,      int32_t* n_reused);

/* ---- Diagnostics -------------------------------------------------------- */
void ion7_print_struct_sizes(void);

/* ---- Context warmup (JIT GPU kernel pre-compilation) ------------------- */
void ion7_context_warmup(struct llama_context* ctx);

/* ---- Advanced sampler (common_sampler) --------------------------------- */
typedef struct {
    uint32_t seed;
    int32_t  top_k;
    float    top_p;
    float    min_p;
    float    xtc_probability;
    float    xtc_threshold;
    float    temp;
    float    repeat_penalty;
    float    freq_penalty;
    float    pres_penalty;
    int32_t  repeat_last_n;
    float    dry_mult;
    float    dry_base;
    int32_t  dry_allowed_len;
    int32_t  dry_last_n;
    int32_t  mirostat;
    float    mirostat_tau;
    float    mirostat_eta;
    int32_t  grammar_lazy;
} ion7_csampler_params_t;

typedef struct ion7_csampler ion7_csampler_t;

ion7_csampler_t* ion7_csampler_init(
    const struct llama_model*     model,
    const ion7_csampler_params_t* params,
    const char*                   grammar,
    const char**                  trigger_words,
    int                           n_triggers,
    const int32_t*                logit_bias_ids,
    const float*                  logit_bias_val,
    int                           n_logit_bias);
void     ion7_csampler_free    (ion7_csampler_t* s);
int32_t  ion7_csampler_sample        (ion7_csampler_t* s, struct llama_context* ctx, int idx, int grammar_first);
void     ion7_csampler_accept        (ion7_csampler_t* s, int32_t token);
int32_t  ion7_csampler_sample_accept (ion7_csampler_t* s, struct llama_context* ctx, int idx, int grammar_first);
void     ion7_csampler_reset   (ion7_csampler_t* s);
int32_t  ion7_csampler_last    (const ion7_csampler_t* s);
uint32_t ion7_csampler_get_seed(const ion7_csampler_t* s);

/* ---- Speculative decoding --------------------------------------------- */
typedef struct ion7_speculative ion7_speculative_t;

ion7_speculative_t* ion7_speculative_init(
    struct llama_context* ctx_tgt,
    struct llama_context* ctx_dft,
    int type, int n_draft, int ngram_min, int ngram_max);
void ion7_speculative_free  (ion7_speculative_t* spec);
void ion7_speculative_begin (ion7_speculative_t* spec, const int32_t* prompt, int n_prompt);
int  ion7_speculative_draft (ion7_speculative_t* spec,
                              const int32_t* prompt, int n_prompt,
                              int32_t last_token,
                              int32_t* out_draft, int max_draft);
void ion7_speculative_accept(ion7_speculative_t* spec, int n_accepted);
void ion7_speculative_stats (const ion7_speculative_t* spec);

/* ---- Chat output parsing ----------------------------------------------- */
int ion7_chat_parse(
    ion7_chat_templates_t* t,
    const char*            text,
    int                    enable_thinking,
    char*   content_buf,  int32_t content_len,
    char*   thinking_buf, int32_t thinking_len,
    char*   tools_buf,    int32_t tools_len,
    int*    out_has_tools);

/* ---- UTF-8 streaming helpers ------------------------------------------- */
int ion7_utf8_seq_len    (uint8_t first_byte);
int ion7_utf8_is_complete(const char* buf, size_t len);

/* ---- JSON Schema → GBNF ----------------------------------------------- */
int ion7_json_schema_to_grammar(const char* schema_json, char* out, size_t out_len);

/* ---- Partial regex matching -------------------------------------------- */
typedef struct ion7_regex ion7_regex_t;

ion7_regex_t* ion7_regex_new   (const char* pattern);
void          ion7_regex_free  (ion7_regex_t* r);
int           ion7_regex_search(ion7_regex_t* r, const char* text, size_t len, int partial);

/* ---- Control vectors (activation steering) ----------------------------- */
int  ion7_cvec_apply(struct llama_context* ctx,
                      const float* data, size_t len,
                      int32_t n_embd, int32_t il_start, int32_t il_end);
void ion7_cvec_clear(struct llama_context* ctx);

/* ---- Tensor inspection (for cb_eval callbacks) ------------------------- */
const char* ion7_tensor_name    (void* t);
int         ion7_tensor_type    (void* t);
int64_t     ion7_tensor_ne      (void* t, int dim);
size_t      ion7_tensor_nbytes  (void* t);
int         ion7_tensor_copy_f32(void* t, float* dst, size_t dst_count);

/* ---- NUMA topology ----------------------------------------------------- */
void ion7_numa_init(int strategy);
int  ion7_is_numa  (void);

/* ---- CPU capability detection ----------------------------------------- */
typedef struct {
    int sse3, ssse3, avx, avx2, avx_vnni, bmi2, f16c, fma;
    int avx512, avx512_vbmi, avx512_vnni, avx512_bf16, amx_int8;
    int neon, arm_fma, fp16_va, dotprod, matmul_int8;
    int sve, sve_cnt, sme;
    int riscv_v, rvv_vlen;
    int vsx, wasm_simd;
} ion7_cpu_caps_t;

void ion7_cpu_caps(ion7_cpu_caps_t* out);

/* ---- Log routing ------------------------------------------------------- */
void ion7_log_to_file      (const char* path);
void ion7_log_set_timestamps(int enable);

/* ---- Base64 ------------------------------------------------------------ */
int ion7_base64_encode(const uint8_t* data, size_t len, char* out, size_t out_len);
int ion7_base64_decode(const char* src, size_t src_len, uint8_t* out, size_t out_len);

/* ---- JSON utilities ---------------------------------------------------- */
int ion7_json_validate(const char* json_str);
int ion7_json_format  (const char* json_str, char* out, size_t out_len);
int ion7_json_merge   (const char* base, const char* overlay, char* out, size_t out_len);

/* ---- Logprob / Entropy (C-side, ~50x faster than Lua for n_vocab loop) - */
float ion7_logprob(struct llama_context* ctx, int32_t idx, int32_t token_id);
float ion7_entropy(struct llama_context* ctx, int32_t idx);
void  ion7_logprob_entropy(struct llama_context* ctx, int32_t idx, int32_t token_id, float* out_logprob, float* out_entropy);
]]

-- ── Internal ─────────────────────────────────────────────────────────────────

local function resolve(opt, env, default)
    if opt  and opt  ~= "" then return opt  end
    local e = os.getenv(env)
    if e and e ~= ""  then return e   end
    return default
end

local function load_lib(path, label)
    local ok, lib = pcall(ffi.load, path)
    if not ok then
        error(string.format("[ion7-core] cannot load %s from '%s': %s",
            label, path, tostring(lib)), 3)
    end
    return lib
end

-- ── Loader ───────────────────────────────────────────────────────────────────

--- @class Loader
local Loader  = {}
Loader.__index = Loader
local _instance

--- Initialize the FFI loader. Idempotent -- safe to call multiple times.
---
--- @param  opts  table?
---   opts.llama_lib   string?  Path to libllama.so
---   opts.bridge_lib  string?  Path to ion7_bridge.so
---   opts.log_level   number?  0=silent 1=error(default) 2=warn 3=info
--- @return table  { lib, bridge, ffi, C }
--- @error  If either library cannot be loaded.
function Loader.init(opts)
    if _instance then return _instance end
    opts = opts or {}

    -- Source directory of this file - default root for bridge .so path resolution.
    local sdir = debug.getinfo(1,"S").source:match("@(.+)/src/") or "."
    -- Default libllama.so: first match in common system paths, nil when absent.
    local def_llama = (function()
        for _, p in ipairs({
            "/usr/local/lib/libllama.so",
            "/usr/lib/libllama.so",
            "/usr/local/lib/libllama.dylib",  -- macOS
        }) do
            local f = io.open(p, "rb")
            if f then f:close(); return p end
        end
        -- No system install found - caller must set LLAMA_LIB or opts.llama_path
        return nil
    end)()

    local llama_path  = resolve(opts.llama_lib or opts.llama_path, "LLAMA_LIB", def_llama)
    if not llama_path or llama_path == "" then
        error("[ion7-core] libllama.so not found.\n" ..
              "  Set LLAMA_LIB=/path/to/libllama.so  or\n" ..
              "  pass opts.llama_path to ion7.init()", 2)
    end
    local bridge_path = resolve(opts.bridge_lib or opts.bridge_path, "ION7_BRIDGE", sdir .. "/bridge/ion7_bridge.so")

    local lib    = load_lib(llama_path,  "libllama.so")
    local bridge = load_lib(bridge_path, "ion7_bridge.so")

    -- ion7_backend_init() belongs to ion7.init() - one-time global init, not per-load.
    bridge.ion7_set_log_level(opts.log_level or 1)

    _instance = setmetatable({
        lib    = lib,
        bridge = bridge,
        ffi    = ffi,
        C      = types.constants,
    }, Loader)

    return _instance
end

--- Return the singleton loader instance. Must call init() first.
--- @return table
--- @error  If init() has not been called.
function Loader.instance()
    assert(_instance, "[ion7-core] Loader.init() must be called before Loader.instance()")
    return _instance
end

--- Free global backend resources. Call at process exit.
--- Also resets the singleton so a new Loader.init() can be called.
--- Useful in tests that need isolated environments.
function Loader.reset()
    if _instance then
        pcall(function() _instance.bridge.ion7_backend_free() end)
        _instance = nil
    end
end

function Loader.shutdown()
    if _instance then
        _instance.bridge.ion7_backend_free()
        _instance = nil
    end
end

return Loader
