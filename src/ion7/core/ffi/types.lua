--- @module ion7.core.ffi.types
--- SPDX-License-Identifier: AGPL-3.0-or-later
---
--- LuaJIT FFI declarations for the llama.cpp public C API.
--- Covers the non-deprecated public surface of llama.h (verified April 2026).
--- Load via ion7.core.ffi.loader, not directly.

local ffi = require "ffi"

ffi.cdef [[

/* ---- Primitives --------------------------------------------------------- */

typedef int32_t llama_token;
typedef int32_t llama_pos;
typedef int32_t llama_seq_id;

/* ---- Enums -------------------------------------------------------------- */

enum llama_vocab_type {
    LLAMA_VOCAB_TYPE_NONE   = 0,
    LLAMA_VOCAB_TYPE_SPM    = 1,
    LLAMA_VOCAB_TYPE_BPE    = 2,
    LLAMA_VOCAB_TYPE_WPM    = 3,
    LLAMA_VOCAB_TYPE_UGM    = 4,
    LLAMA_VOCAB_TYPE_RWKV   = 5,
    LLAMA_VOCAB_TYPE_PLAMO2 = 6,
};

enum llama_rope_type {
    LLAMA_ROPE_TYPE_NONE   = -1,
    LLAMA_ROPE_TYPE_NORM   =  0,
    LLAMA_ROPE_TYPE_NEOX   =  2,
    LLAMA_ROPE_TYPE_MROPE  = 10,
    LLAMA_ROPE_TYPE_IMROPE = 24,
};

enum llama_token_attr {
    LLAMA_TOKEN_ATTR_UNDEFINED    = 0,
    LLAMA_TOKEN_ATTR_UNKNOWN      = 1,
    LLAMA_TOKEN_ATTR_UNUSED       = 2,
    LLAMA_TOKEN_ATTR_NORMAL       = 4,
    LLAMA_TOKEN_ATTR_CONTROL      = 8,
    LLAMA_TOKEN_ATTR_USER_DEFINED = 16,
    LLAMA_TOKEN_ATTR_BYTE         = 32,
    LLAMA_TOKEN_ATTR_NORMALIZED   = 64,
    LLAMA_TOKEN_ATTR_LSTRIP       = 128,
    LLAMA_TOKEN_ATTR_RSTRIP       = 256,
    LLAMA_TOKEN_ATTR_SINGLE_WORD  = 512,
};

enum llama_pooling_type {
    LLAMA_POOLING_TYPE_UNSPECIFIED = -1,
    LLAMA_POOLING_TYPE_NONE        =  0,
    LLAMA_POOLING_TYPE_MEAN        =  1,
    LLAMA_POOLING_TYPE_CLS         =  2,
    LLAMA_POOLING_TYPE_LAST        =  3,
    LLAMA_POOLING_TYPE_RANK        =  4,
};

/* llama_ftype : quantization formats */
enum llama_ftype {
    LLAMA_FTYPE_ALL_F32        = 0,
    LLAMA_FTYPE_MOSTLY_F16     = 1,
    LLAMA_FTYPE_MOSTLY_Q4_0    = 2,
    LLAMA_FTYPE_MOSTLY_Q4_1    = 3,
    LLAMA_FTYPE_MOSTLY_Q8_0    = 7,
    LLAMA_FTYPE_MOSTLY_Q5_0    = 8,
    LLAMA_FTYPE_MOSTLY_Q5_1    = 9,
    LLAMA_FTYPE_MOSTLY_Q2_K    = 10,
    LLAMA_FTYPE_MOSTLY_Q3_K_S  = 11,
    LLAMA_FTYPE_MOSTLY_Q3_K_M  = 12,
    LLAMA_FTYPE_MOSTLY_Q3_K_L  = 13,
    LLAMA_FTYPE_MOSTLY_Q4_K_S  = 14,
    LLAMA_FTYPE_MOSTLY_Q4_K_M  = 15,
    LLAMA_FTYPE_MOSTLY_Q5_K_S  = 16,
    LLAMA_FTYPE_MOSTLY_Q5_K_M  = 17,
    LLAMA_FTYPE_MOSTLY_Q6_K    = 18,
    LLAMA_FTYPE_MOSTLY_IQ4_XS  = 30,
    LLAMA_FTYPE_MOSTLY_BF16    = 32,
    LLAMA_FTYPE_GUESSED        = 1024,
};

enum llama_rope_scaling_type {
    LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1,
    LLAMA_ROPE_SCALING_TYPE_NONE        =  0,
    LLAMA_ROPE_SCALING_TYPE_LINEAR      =  1,
    LLAMA_ROPE_SCALING_TYPE_YARN        =  2,
    LLAMA_ROPE_SCALING_TYPE_LONGROPE    =  3,
};

enum llama_attention_type {
    LLAMA_ATTENTION_TYPE_UNSPECIFIED = -1,
    LLAMA_ATTENTION_TYPE_CAUSAL      =  0,
    LLAMA_ATTENTION_TYPE_NON_CAUSAL  =  1,
};

enum llama_flash_attn_type {
    LLAMA_FLASH_ATTN_TYPE_AUTO     = -1,
    LLAMA_FLASH_ATTN_TYPE_DISABLED =  0,
    LLAMA_FLASH_ATTN_TYPE_ENABLED  =  1,
};

enum llama_split_mode {
    LLAMA_SPLIT_MODE_NONE  = 0,
    LLAMA_SPLIT_MODE_LAYER = 1,
    LLAMA_SPLIT_MODE_ROW   = 2,
};

enum llama_params_fit_status {
    LLAMA_PARAMS_FIT_STATUS_SUCCESS = 0,
    LLAMA_PARAMS_FIT_STATUS_FAILURE = 1,
    LLAMA_PARAMS_FIT_STATUS_ERROR   = 2,
};

/* ggml types used for KV cache quantization */
enum ggml_type {
    GGML_TYPE_F32   =  0,
    GGML_TYPE_F16   =  1,
    GGML_TYPE_Q4_0  =  2,
    GGML_TYPE_Q4_1  =  3,
    GGML_TYPE_Q5_0  =  6,
    GGML_TYPE_Q5_1  =  7,
    GGML_TYPE_Q8_0  =  8,
    GGML_TYPE_Q4_K  = 12,
    GGML_TYPE_Q5_K  = 13,
    GGML_TYPE_Q6_K  = 14,
    GGML_TYPE_IQ4_NL = 20,
    GGML_TYPE_BF16  = 30,
};

enum ggml_log_level {
    GGML_LOG_LEVEL_NONE  = 0,
    GGML_LOG_LEVEL_DEBUG = 1,
    GGML_LOG_LEVEL_INFO  = 2,
    GGML_LOG_LEVEL_WARN  = 3,
    GGML_LOG_LEVEL_ERROR = 4,
    GGML_LOG_LEVEL_CONT  = 5,
};

/* ---- Opaque handles ----------------------------------------------------- */

typedef struct llama_model         llama_model;
typedef struct llama_context       llama_context;
typedef struct llama_vocab         llama_vocab;
typedef struct llama_sampler       llama_sampler;
typedef struct llama_adapter_lora  llama_adapter_lora;
typedef void*                      llama_memory_t;

/* ---- llama_batch -------------------------------------------------------- */

typedef struct llama_batch {
    int32_t        n_tokens;
    llama_token*   token;
    float*         embd;
    llama_pos*     pos;
    int32_t*       n_seq_id;
    llama_seq_id** seq_id;
    int8_t*        logits;
} llama_batch;

/* ---- llama_token_data -------------------------------------------------- */

typedef struct llama_token_data {
    llama_token id;
    float       logit;
    float       p;
} llama_token_data;

typedef struct llama_token_data_array {
    llama_token_data* data;
    size_t            size;
    int64_t           selected;
    bool              sorted;
} llama_token_data_array;

/* ---- llama_logit_bias -------------------------------------------------- */

/* Quantization parameters */
typedef struct llama_model_quantize_params {
    int32_t  nthread;
    enum llama_ftype ftype;
    enum ggml_type output_tensor_type;
    enum ggml_type token_embedding_type;
    bool     allow_requantize;
    bool     quantize_output_tensor;
    bool     only_copy;
    bool     pure;
    bool     keep_split;
    bool     dry_run;
    const void* imatrix;
    const void* kv_overrides;
    const void* tt_overrides;
    const int32_t* prune_layers;
} llama_model_quantize_params;

typedef struct llama_logit_bias {
    llama_token token;
    float       bias;
} llama_logit_bias;

/* ---- llama_chat_message ------------------------------------------------ */

typedef struct llama_chat_message {
    const char* role;
    const char* content;
} llama_chat_message;

/* ---- llama_sampler_chain_params --------------------------------------- */

typedef struct llama_sampler_chain_params {
    bool no_perf;
} llama_sampler_chain_params;

/* ---- Performance data -------------------------------------------------- */

typedef struct llama_perf_context_data {
    double  t_start_ms;
    double  t_load_ms;
    double  t_p_eval_ms;
    double  t_eval_ms;
    int32_t n_p_eval;
    int32_t n_eval;
    int32_t n_reused;  /* number of times a compute graph was reused */
} llama_perf_context_data;

typedef struct llama_perf_sampler_data {
    double  t_sample_ms;
    int32_t n_sample;
} llama_perf_sampler_data;

/* ---- Callbacks ---------------------------------------------------------- */

typedef bool (*llama_progress_callback)(float progress, void* user_data);
typedef void (*ggml_log_callback)(enum ggml_log_level level, const char* text, void* ud);
typedef bool (*ggml_abort_callback)(void* data);
typedef bool (*ggml_backend_sched_eval_callback)(void* t, bool ask, void* ud);

/* ---- Core API ----------------------------------------------------------- */

/* Backend */
void        llama_backend_init(void);
void        llama_backend_free(void);
void        llama_log_set(ggml_log_callback cb, void* ud);
void        llama_log_get(ggml_log_callback* cb, void** ud);
const char* llama_print_system_info(void);
bool        llama_supports_mmap(void);
bool        llama_supports_mlock(void);
bool        llama_supports_gpu_offload(void);
bool        llama_supports_rpc(void);
/* NUMA optimization for multi-socket servers */
void        llama_numa_init(enum ggml_numa_strategy numa);
size_t      llama_max_devices(void);
size_t      llama_max_parallel_sequences(void);
size_t      llama_max_tensor_buft_overrides(void);
int64_t     llama_time_us(void);

/* Model loading */
llama_model* llama_model_load_from_file(const char* path,
    struct {
        void*    devices;
        void*    tensor_buft_overrides;
        int32_t  n_gpu_layers;
        enum llama_split_mode split_mode;
        int32_t  main_gpu;
        const float* tensor_split;
        llama_progress_callback progress_callback;
        void*    progress_callback_user_data;
        const void* kv_overrides;
        bool     vocab_only;
        bool     use_mmap;
        bool     use_direct_io;
        bool     use_mlock;
        bool     check_tensors;
        bool     use_extra_bufts;
        bool     no_host;
        bool     no_alloc;
    } params);
void llama_model_free(llama_model* model);
/* Load a sharded model from multiple GGUF files */
llama_model* llama_model_load_from_splits(const char** paths, size_t n_paths,
    struct {
        void*    devices;
        void*    tensor_buft_overrides;
        int32_t  n_gpu_layers;
        enum llama_split_mode split_mode;
        int32_t  main_gpu;
        const float* tensor_split;
        llama_progress_callback progress_callback;
        void*    progress_callback_user_data;
        const void* kv_overrides;
        bool     vocab_only;
        bool     use_mmap;
        bool     use_direct_io;
        bool     use_mlock;
        bool     check_tensors;
        bool     use_extra_bufts;
        bool     no_host;
        bool     no_alloc;
    } params);
void llama_model_save_to_file(const llama_model* model, const char* path);

/* Model introspection */
int32_t              llama_model_n_ctx_train(const llama_model* model);
int32_t              llama_model_n_embd     (const llama_model* model);
int32_t              llama_model_n_embd_inp (const llama_model* model);
int32_t              llama_model_n_embd_out (const llama_model* model);
int32_t              llama_model_n_layer    (const llama_model* model);
int32_t              llama_model_n_head     (const llama_model* model);
int32_t              llama_model_n_head_kv  (const llama_model* model);
int32_t              llama_model_n_swa      (const llama_model* model);
uint64_t             llama_model_size       (const llama_model* model);
uint64_t             llama_model_n_params   (const llama_model* model);
bool                 llama_model_has_encoder(const llama_model* model);
bool                 llama_model_has_decoder(const llama_model* model);
llama_token          llama_model_decoder_start_token(const llama_model* model);
bool                 llama_model_is_recurrent(const llama_model* model);
bool                 llama_model_is_hybrid   (const llama_model* model);
bool                 llama_model_is_diffusion(const llama_model* model);
int32_t              llama_model_desc        (const llama_model* model, char* buf, size_t sz);
const char*          llama_model_chat_template(const llama_model* model, const char* name);
float                llama_model_rope_freq_scale_train(const llama_model* model);
enum llama_rope_type llama_model_rope_type  (const llama_model* model);
uint32_t             llama_model_n_cls_out  (const llama_model* model);
const char*          llama_model_cls_label  (const llama_model* model, uint32_t i);

/* Model metadata */
int32_t llama_model_meta_val_str        (const llama_model* m, const char* key, char* buf, size_t sz);
int32_t llama_model_meta_count          (const llama_model* m);
int32_t llama_model_meta_key_by_index   (const llama_model* m, int32_t i, char* buf, size_t sz);
int32_t llama_model_meta_val_str_by_index(const llama_model* m, int32_t i, char* buf, size_t sz);

/* Vocab */
const llama_vocab* llama_model_get_vocab      (const llama_model* model);
int32_t            llama_vocab_n_tokens       (const llama_vocab* vocab);
enum llama_vocab_type llama_vocab_type        (const llama_vocab* vocab);
const char*        llama_vocab_get_text       (const llama_vocab* vocab, llama_token token);
float              llama_vocab_get_score      (const llama_vocab* vocab, llama_token token);
enum llama_token_attr llama_vocab_get_attr    (const llama_vocab* vocab, llama_token token);
bool               llama_vocab_is_eog         (const llama_vocab* vocab, llama_token token);
bool               llama_vocab_is_control     (const llama_vocab* vocab, llama_token token);
llama_token        llama_vocab_bos            (const llama_vocab* vocab);
llama_token        llama_vocab_eos            (const llama_vocab* vocab);
llama_token        llama_vocab_eot            (const llama_vocab* vocab);
llama_token        llama_vocab_sep            (const llama_vocab* vocab);
llama_token        llama_vocab_nl             (const llama_vocab* vocab);
llama_token        llama_vocab_pad            (const llama_vocab* vocab);
llama_token        llama_vocab_mask           (const llama_vocab* vocab);
llama_token        llama_vocab_cls            (const llama_vocab* vocab);
bool               llama_vocab_get_add_bos    (const llama_vocab* vocab);
bool               llama_vocab_get_add_eos    (const llama_vocab* vocab);
bool               llama_vocab_get_add_sep    (const llama_vocab* vocab);
llama_token        llama_vocab_fim_pre        (const llama_vocab* vocab);
llama_token        llama_vocab_fim_suf        (const llama_vocab* vocab);
llama_token        llama_vocab_fim_mid        (const llama_vocab* vocab);
llama_token        llama_vocab_fim_pad        (const llama_vocab* vocab);
llama_token        llama_vocab_fim_rep        (const llama_vocab* vocab);
llama_token        llama_vocab_fim_sep        (const llama_vocab* vocab);

/* Tokenization */
int32_t llama_tokenize         (const llama_vocab* vocab, const char* text, int32_t text_len, llama_token* tokens, int32_t n_max, bool add_special, bool parse_special);
int32_t llama_token_to_piece   (const llama_vocab* vocab, llama_token token, char* buf, int32_t length, int32_t lstrip, bool special);
int32_t llama_detokenize       (const llama_vocab* vocab, const llama_token* tokens, int32_t n, char* text, int32_t text_len_max, bool remove_special, bool unparse_special);
/* Use ion7_chat_apply_template() from the bridge instead of this function. */
int32_t llama_chat_builtin_templates(const char** output, size_t len);

/* Context */
llama_context* llama_init_from_model(llama_model* model,
    struct {
        uint32_t n_ctx;
        uint32_t n_batch;
        uint32_t n_ubatch;
        uint32_t n_seq_max;
        int32_t  n_threads;
        int32_t  n_threads_batch;
        enum llama_rope_scaling_type rope_scaling_type;
        enum llama_pooling_type      pooling_type;
        enum llama_attention_type    attention_type;
        enum llama_flash_attn_type   flash_attn_type;
        float    rope_freq_base;
        float    rope_freq_scale;
        float    yarn_ext_factor;
        float    yarn_attn_factor;
        float    yarn_beta_fast;
        float    yarn_beta_slow;
        uint32_t yarn_orig_ctx;
        float    defrag_thold;
        ggml_backend_sched_eval_callback cb_eval;
        void*    cb_eval_user_data;
        enum ggml_type type_k;
        enum ggml_type type_v;
        ggml_abort_callback abort_callback;
        void*    abort_callback_data;
        bool     embeddings;
        bool     offload_kqv;
        bool     no_perf;
        bool     op_offload;
        bool     swa_full;
        bool     kv_unified;
        void*    samplers;
        size_t   n_samplers;
    } params);
void     llama_free          (llama_context* ctx);
uint32_t llama_n_ctx         (const llama_context* ctx);
uint32_t llama_n_ctx_seq     (const llama_context* ctx);
uint32_t llama_n_batch       (const llama_context* ctx);
uint32_t llama_n_ubatch      (const llama_context* ctx);
uint32_t llama_n_seq_max     (const llama_context* ctx);
int32_t  llama_n_threads     (llama_context* ctx);
int32_t  llama_n_threads_batch(llama_context* ctx);
void     llama_set_n_threads (llama_context* ctx, int32_t n_threads, int32_t n_threads_batch);
void     llama_set_embeddings(llama_context* ctx, bool embeddings);
void     llama_set_causal_attn(llama_context* ctx, bool causal_attn);
void     llama_set_warmup    (llama_context* ctx, bool warmup);
void     llama_set_abort_callback(llama_context* ctx, ggml_abort_callback cb, void* data);
void     llama_synchronize   (llama_context* ctx);
enum llama_pooling_type llama_get_pooling_type(const llama_context* ctx);

/* Context queries */
const llama_model* llama_get_model  (const llama_context* ctx);
llama_memory_t     llama_get_memory (const llama_context* ctx);

/* Batch */
llama_batch llama_batch_get_one(const llama_token* tokens, int32_t n_tokens);
llama_batch llama_batch_init   (int32_t n_tokens, int32_t embd, int32_t n_seq_max);
void        llama_batch_free   (llama_batch batch);

/* Decode / Encode */
int32_t llama_decode(llama_context* ctx, llama_batch batch);
int32_t llama_encode(llama_context* ctx, llama_batch batch);

/* Logits / Embeddings */
float* llama_get_logits        (llama_context* ctx);
float* llama_get_logits_ith    (llama_context* ctx, int32_t i);
float* llama_get_embeddings    (llama_context* ctx);
float* llama_get_embeddings_ith(llama_context* ctx, int32_t i);
float* llama_get_embeddings_seq(llama_context* ctx, llama_seq_id seq_id);

/* Per-position sampling results (inspect what was sampled after decode+sample) */
llama_token llama_get_sampled_token_ith            (llama_context* ctx, int32_t i);
float*      llama_get_sampled_probs_ith            (llama_context* ctx, int32_t i);
uint32_t    llama_get_sampled_probs_count_ith      (llama_context* ctx, int32_t i);
float*      llama_get_sampled_logits_ith           (llama_context* ctx, int32_t i);
uint32_t    llama_get_sampled_logits_count_ith     (llama_context* ctx, int32_t i);
llama_token* llama_get_sampled_candidates_ith      (llama_context* ctx, int32_t i);
uint32_t     llama_get_sampled_candidates_count_ith(llama_context* ctx, int32_t i);

/* Memory / KV cache */
void      llama_memory_clear        (llama_memory_t mem, bool data);
bool      llama_memory_seq_rm       (llama_memory_t mem, llama_seq_id seq, llama_pos p0, llama_pos p1);
void      llama_memory_seq_cp       (llama_memory_t mem, llama_seq_id src, llama_seq_id dst, llama_pos p0, llama_pos p1);
void      llama_memory_seq_keep     (llama_memory_t mem, llama_seq_id seq);
void      llama_memory_seq_add      (llama_memory_t mem, llama_seq_id seq, llama_pos p0, llama_pos p1, llama_pos delta);
void      llama_memory_seq_div      (llama_memory_t mem, llama_seq_id seq, llama_pos p0, llama_pos p1, int d);
llama_pos llama_memory_seq_pos_min  (llama_memory_t mem, llama_seq_id seq);
llama_pos llama_memory_seq_pos_max  (llama_memory_t mem, llama_seq_id seq);
bool      llama_memory_can_shift    (llama_memory_t mem);

/* Context state (full) */
size_t llama_state_get_size (llama_context* ctx);
size_t llama_state_get_data (llama_context* ctx, uint8_t* dst, size_t size);
size_t llama_state_set_data (llama_context* ctx, const uint8_t* src, size_t size);
bool   llama_state_save_file(llama_context* ctx, const char* path, const llama_token* tokens, size_t n);
bool   llama_state_load_file(llama_context* ctx, const char* path, llama_token* out, size_t cap, size_t* n_out);

/* Per-sequence state flags (bitfield) */
typedef uint32_t llama_state_seq_flags;

/* Per-sequence state */
size_t llama_state_seq_get_size (llama_context* ctx, llama_seq_id seq_id);
size_t llama_state_seq_get_data (llama_context* ctx, uint8_t* dst, size_t size, llama_seq_id seq_id);
size_t llama_state_seq_set_data (llama_context* ctx, const uint8_t* src, size_t size, llama_seq_id seq_id);
size_t llama_state_seq_save_file(llama_context* ctx, const char* path, llama_seq_id seq_id, const llama_token* tokens, size_t n_token_count);
size_t llama_state_seq_load_file(llama_context* ctx, const char* path, llama_seq_id dest_seq_id, llama_token* tokens_out, size_t n_token_capacity, size_t* n_token_count_out);
size_t llama_state_seq_get_size_ext(llama_context* ctx, llama_seq_id seq_id, llama_state_seq_flags flags);
size_t llama_state_seq_get_data_ext(llama_context* ctx, uint8_t* dst, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags);
size_t llama_state_seq_set_data_ext(llama_context* ctx, const uint8_t* src, size_t size, llama_seq_id dest_seq_id, llama_state_seq_flags flags);

/* LoRA */
llama_adapter_lora* llama_adapter_lora_init(llama_model* model, const char* path_lora);
void                llama_adapter_lora_free(llama_adapter_lora* adapter);
int32_t             llama_adapter_meta_val_str        (const llama_adapter_lora* a, const char* key, char* buf, size_t sz);
int32_t             llama_adapter_meta_count          (const llama_adapter_lora* a);
int32_t             llama_adapter_meta_key_by_index   (const llama_adapter_lora* a, int32_t i, char* buf, size_t sz);
int32_t             llama_adapter_meta_val_str_by_index(const llama_adapter_lora* a, int32_t i, char* buf, size_t sz);

/* Control vectors (steering) */
int32_t llama_set_adapter_cvec(llama_context* ctx, const float* data, size_t len, int32_t n_embd, int32_t il_start, int32_t il_end);

/* Sampler chain */
llama_sampler_chain_params llama_sampler_chain_default_params(void);
llama_sampler* llama_sampler_chain_init  (llama_sampler_chain_params params);
void           llama_sampler_chain_add   (llama_sampler* chain, llama_sampler* smpl);
llama_sampler* llama_sampler_chain_get   (llama_sampler* chain, int32_t i);
int            llama_sampler_chain_n     (const llama_sampler* chain);
llama_sampler* llama_sampler_chain_remove(llama_sampler* chain, int32_t i);

/* Sampler operations */
const char*    llama_sampler_name  (const llama_sampler* smpl);
void           llama_sampler_accept(llama_sampler* smpl, llama_token token);
void           llama_sampler_apply (llama_sampler* smpl, llama_token_data_array* cur_p);
void           llama_sampler_reset (llama_sampler* smpl);
llama_sampler* llama_sampler_clone (const llama_sampler* smpl);
void           llama_sampler_free  (llama_sampler* smpl);
llama_token    llama_sampler_sample (llama_sampler* smpl, llama_context* ctx, int32_t idx);
uint32_t       llama_sampler_get_seed(const llama_sampler* smpl);

/* Custom sampler interface -- implement in Lua via ffi.cast() callbacks */
typedef void* llama_sampler_context_t;

struct llama_sampler_i {
    const char*    (*name)  (const struct llama_sampler* smpl);
    void           (*accept)(struct llama_sampler* smpl, int32_t token);
    void           (*apply) (struct llama_sampler* smpl, struct llama_token_data_array* cur_p);
    void           (*reset) (struct llama_sampler* smpl);
    struct llama_sampler* (*clone)(const struct llama_sampler* smpl);
    void           (*free)  (struct llama_sampler* smpl);
    bool (*backend_init)    (struct llama_sampler* smpl, void* buft);
    void (*backend_accept)  (struct llama_sampler* smpl, void* ctx, void* gf, void* tok);
    void (*backend_apply)   (struct llama_sampler* smpl, void* ctx, void* gf, void* data);
    void (*backend_set_input)(struct llama_sampler* smpl);
};

/* Create a sampler from a custom llama_sampler_i interface */
llama_sampler* llama_sampler_init(struct llama_sampler_i* iface, llama_sampler_context_t ctx);

/* Individual samplers */
llama_sampler* llama_sampler_init_greedy     (void);
llama_sampler* llama_sampler_init_dist       (uint32_t seed);
llama_sampler* llama_sampler_init_top_k      (int32_t k);
llama_sampler* llama_sampler_init_top_p      (float p, size_t min_keep);
llama_sampler* llama_sampler_init_min_p      (float p, size_t min_keep);
llama_sampler* llama_sampler_init_typical    (float p, size_t min_keep);
llama_sampler* llama_sampler_init_temp       (float t);
llama_sampler* llama_sampler_init_temp_ext   (float t, float delta, float exponent);
llama_sampler* llama_sampler_init_top_n_sigma(float n);
llama_sampler* llama_sampler_init_xtc        (float prob, float threshold, size_t min_keep, uint32_t seed);
llama_sampler* llama_sampler_init_mirostat   (int32_t n_vocab, uint32_t seed, float tau, float eta, int32_t m);
llama_sampler* llama_sampler_init_mirostat_v2(uint32_t seed, float tau, float eta);
llama_sampler* llama_sampler_init_grammar    (const llama_vocab* vocab, const char* grammar_str, const char* grammar_root);
llama_sampler* llama_sampler_init_grammar_lazy_patterns(const llama_vocab* vocab, const char* grammar_str, const char* grammar_root, const char** trigger_words, size_t num_trigger_words, const llama_token* trigger_tokens, size_t num_trigger_tokens, const char** trigger_patterns, size_t num_trigger_patterns);
llama_sampler* llama_sampler_init_penalties  (int32_t penalty_last_n, float penalty_repeat, float penalty_freq, float penalty_present);
llama_sampler* llama_sampler_init_dry        (const llama_vocab* vocab, int32_t n_ctx_train, float dry_multiplier, float dry_base, int32_t dry_allowed_length, int32_t dry_penalty_last_n, const char** seq_breakers, size_t num_breakers);
llama_sampler* llama_sampler_init_adaptive_p (float target, float decay, uint32_t seed);
llama_sampler* llama_sampler_init_logit_bias (int32_t n_vocab, int32_t n_logit_bias, const llama_logit_bias* logit_bias);
llama_sampler* llama_sampler_init_infill     (const llama_vocab* vocab);

/* Performance */
llama_perf_context_data llama_perf_context      (const llama_context* ctx);
void                    llama_perf_context_print (const llama_context* ctx);
void                    llama_perf_context_reset (llama_context* ctx);
llama_perf_sampler_data llama_perf_sampler      (const llama_sampler* chain);
void                    llama_perf_sampler_print (const llama_sampler* chain);
void                    llama_perf_sampler_reset (llama_sampler* chain);
void                    llama_memory_breakdown_print(const llama_context* ctx);

/* Model quantize */
llama_model_quantize_params llama_model_quantize_default_params(void);
uint32_t llama_model_quantize(const char* fname_inp, const char* fname_out,
                              const llama_model_quantize_params* params);

/* Load from FILE* */
llama_model* llama_model_load_from_file_ptr(void* file,
    struct {
        void*    devices;
        void*    tensor_buft_overrides;
        int32_t  n_gpu_layers;
        enum llama_split_mode split_mode;
        int32_t  main_gpu;
        const float* tensor_split;
        llama_progress_callback progress_callback;
        void*    progress_callback_user_data;
        const void* kv_overrides;
        bool     vocab_only;
        bool     use_mmap;
        bool     use_direct_io;
        bool     use_mlock;
        bool     check_tensors;
        bool     use_extra_bufts;
        bool     no_host;
        bool     no_alloc;
    } params);

/* Per-sequence sampler (experimental) */
bool llama_set_sampler(llama_context* ctx, llama_seq_id seq_id, llama_sampler* smpl);

/* ALora (adaptive LoRA) invocation tokens */
uint64_t            llama_adapter_get_alora_n_invocation_tokens(const llama_adapter_lora* adapter);
const llama_token*  llama_adapter_get_alora_invocation_tokens  (const llama_adapter_lora* adapter);

/* Model meta key string from enum */
const char* llama_model_meta_key_str(int key);

/* Utilities */
int32_t     llama_split_path  (char* split_path, size_t maxlen, const char* path_prefix, int32_t split_no, int32_t split_count);
int32_t     llama_split_prefix(char* split_prefix, size_t maxlen, const char* split_path, int32_t split_no, int32_t split_count);
const char* llama_flash_attn_type_name(enum llama_flash_attn_type t);

]]

return {
    ffi       = ffi,
    constants = {
        TOKEN_NULL      = -1,
        POOLING_NONE    = 0,
        POOLING_MEAN    = 1,
        POOLING_CLS     = 2,
        POOLING_LAST    = 3,
        POOLING_RANK    = 4,
        KV_F16          =  1,   -- GGML_TYPE_F16
        KV_BF16         = 30,   -- GGML_TYPE_BF16
        KV_Q8_0         =  8,   -- GGML_TYPE_Q8_0
        KV_Q4_0         =  2,   -- GGML_TYPE_Q4_0
        KV_Q4_1         =  3,   -- GGML_TYPE_Q4_1
        KV_Q5_0         =  6,   -- GGML_TYPE_Q5_0
        KV_Q5_1         =  7,   -- GGML_TYPE_Q5_1
        KV_IQ4_NL       = 20,   -- GGML_TYPE_IQ4_NL
        LOG_SILENT      = 0,
        LOG_ERROR       = 1,
        LOG_WARN        = 2,
        LOG_INFO        = 3,
        LOG_DEBUG       = 4,
        FIT_SUCCESS     = 0,
        FIT_FAILURE     = 1,
        FIT_ERROR       = 2,
        ROPE_NONE       = -1,
        ROPE_NORM       = 0,
        ROPE_NEOX       = 2,
        ROPE_MROPE      = 10,
        ROPE_IMROPE     = 24,
    },
}
