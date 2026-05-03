/*
 * ion7_bridge.h - Public C API of the ion7-core bridge.
 *
 * Copyright (C) 2026 Ion7 Project Contributors
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
 *
 * ──────────────────────────────────────────────────────────────────────────
 * SCOPE - ion7-core bridge v2.0
 *
 * Starting with ion7-core 2.0 the bridge wraps ONE thing only:
 * llama.cpp's `common/` C++ helper layer (libcommon).
 *
 * Everything that lives in `llama.h` / `ggml*.h` is now consumed directly
 * by Lua via the auto-generated FFI bindings under `src/ion7/core/ffi/`.
 * Re-introducing pass-through wrappers here would only add maintenance
 * cost without any value.
 *
 * Bridge-specific logic that does not need libcommon (UTF-8 helpers,
 * base64, log routing, logprob/entropy, tensor inspection, context
 * warmup) lives in pure-Lua modules under `src/ion7/core/`.
 *
 * What this header still exposes:
 *   - VRAM auto-fit                      (libcommon: llama_params_fit)
 *   - Jinja2 chat templates              (libcommon: common_chat_templates)
 *   - Chat output parsing                (libcommon: common_chat_parse)
 *   - Reasoning-budget sampler           (libcommon: common_reasoning_budget)
 *   - common_sampler                     (DRY, XTC, Mirostat, grammar_lazy)
 *   - common_speculative                 (n-gram, draft model, EAGLE-3)
 *   - JSON Schema → GBNF                 (libcommon: json_schema_to_grammar)
 *   - Partial regex                      (libcommon: regex_partial)
 *   - JSON validate / format / merge     (nlohmann/json — already linked)
 *   - llama_opt training loop            (libcommon: common_opt_dataset_init)
 *
 * ──────────────────────────────────────────────────────────────────────────
 * STABILITY CONTRACT
 *
 * This API is stable across minor versions. Breaking changes require a
 * major version bump (3.0). Direct llama.cpp / ggml symbols accessed via
 * FFI are NOT covered by this contract — see `src/ion7/core/compat/` for
 * version-aware shims when llama.cpp upstream renames things.
 *
 * Naming convention : ion7_<noun>_<verb>
 * Linkage           : pure C — no C++, no STL, no exceptions cross the ABI.
 * Parameters        : primitive C types only (int, float, char*, size_t).
 * ──────────────────────────────────────────────────────────────────────────
 */

#ifndef ION7_BRIDGE_H
#define ION7_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

#include "llama.h"
#include "ggml-opt.h"

/* =========================================================================
 * JIT-friendly decode / encode shims (single exception to the
 * "libcommon-only" rule documented above)
 *
 * Upstream `llama_decode` / `llama_encode` take `struct llama_batch` by
 * VALUE. LuaJIT's FFI cannot JIT-compile C calls with aggregates passed
 * by value — every per-token call aborts the trace, drops back to the
 * interpreter, and re-stitches on return. These shims forward by
 * pointer so the Lua generation loop compiles into a single trace.
 *
 * Behaviourally identical to the upstream functions ; semantics, error
 * codes, KV cache effects are unchanged.
 * ======================================================================= */

/** Pointer-passing wrapper around `llama_decode`. */
int32_t ion7_context_decode(struct llama_context * ctx, const struct llama_batch * batch);

/** Pointer-passing wrapper around `llama_encode`. */
int32_t ion7_context_encode(struct llama_context * ctx, const struct llama_batch * batch);

/* =========================================================================
 * Version
 * ======================================================================= */

/** Returns the bridge version string (semver), e.g. `"2.0.0"`. */
const char* ion7_bridge_version(void);

/* =========================================================================
 * VRAM auto-fit
 * ======================================================================= */

/**
 * Probe available device memory and compute the maximum `n_gpu_layers`
 * and `n_ctx` that fit without OOM. Wraps `llama_params_fit` from
 * libcommon (which lives in `common/fit.cpp`, not in `llama.h`).
 *
 * Not thread-safe. Call before loading any model.
 *
 * @param  path          Path to the GGUF file (used for size estimation).
 * @param  n_gpu_layers  Output: recommended GPU layer count.
 * @param  n_ctx         In/out: requested context size, may be reduced.
 * @param  n_ctx_min     Minimum acceptable context size; abort if cannot meet.
 * @return               0 = success, 1 = cannot fit, 2 = error.
 */
int ion7_params_fit(const char* path,
                    int32_t* n_gpu_layers,
                    uint32_t* n_ctx,
                    uint32_t  n_ctx_min);

/* =========================================================================
 * Chat Templates (Jinja2 native)
 * ======================================================================= */

/**
 * Opaque handle to the chat-template stack.
 *
 * Internally wraps a `common_chat_templates *` AND a memo of the most
 * recent `common_chat_params` returned by `common_chat_templates_apply`.
 *
 * `ion7_chat_templates_apply` snapshots the resulting `common_chat_params`
 * (format + generation_prompt + serialised PEG parser arena) on every
 * call, and `ion7_chat_parse` consumes that snapshot to feed
 * `common_chat_parser_params` correctly. That is what lets templates
 * with non-default tool-call formats (Mistral `[TOOL_CALLS]…[ARGS]…`,
 * Qwen `<|tool_call_begin|>`, Hermes `<tool_call>…</tool_call>`) be
 * parsed back into structured tool_calls without the caller having
 * to know which family the model belongs to.
 */
typedef struct ion7_chat_templates_priv ion7_chat_templates_t;

/**
 * Initialise chat templates from a loaded model.
 *
 * @param  model          Loaded model whose Jinja2 template will be parsed.
 * @param  tmpl_override  Override the embedded template, or NULL for default.
 * @return                Opaque handle, or NULL on failure.
 *                        Free with `ion7_chat_templates_free`.
 */
ion7_chat_templates_t* ion7_chat_templates_init(
    const struct llama_model* model,
    const char*               tmpl_override);

/** Free a chat templates handle. Safe to call with NULL. */
void ion7_chat_templates_free(ion7_chat_templates_t* t);

/**
 * Returns 1 if the model's template supports the `enable_thinking`
 * parameter (Qwen3, DeepSeek-R1 and similar reasoning models).
 */
int ion7_chat_templates_support_thinking(const ion7_chat_templates_t* t);

/**
 * Apply the Jinja2 template with full parameter support.
 *
 * @param  t                Handle from `ion7_chat_templates_init`.
 * @param  roles            Array of role strings ("system", "user", "assistant").
 * @param  contents         Array of content strings.
 * @param  n_msgs           Number of messages.
 * @param  add_ass          1 = append assistant generation prefix.
 * @param  enable_thinking  -1 = model default, 0 = force off, 1 = force on.
 * @param  buf              Output buffer; may be NULL for size query.
 * @param  buf_len          Capacity of `buf` in bytes.
 * @return                  Total bytes needed (incl. NUL), -1 if `t` is NULL,
 *                          -2 on template error, -3 if the prompt exceeds 2 GB.
 *                          If the return value is greater than `buf_len`,
 *                          resize the buffer and retry.
 */
int32_t ion7_chat_templates_apply(
    ion7_chat_templates_t* t,
    const char**           roles,
    const char**           contents,
    size_t                 n_msgs,
    int                    add_ass,
    int                    enable_thinking,
    char*                  buf,
    int32_t                buf_len);

/* =========================================================================
 * Chat output parsing (tool calls + thinking extraction)
 * ======================================================================= */

/**
 * Parse raw model output into structured content, thinking block, and a
 * JSON array of tool calls. Buffers that are too small are NUL-terminated
 * at their capacity limit.
 *
 * @param  t                Chat templates handle.
 * @param  text             Raw model output (NUL-terminated).
 * @param  enable_thinking  1 = thinking enabled, 0 = disabled, -1 = auto.
 * @param  content_buf      Out: main response text.
 * @param  content_len      Capacity of `content_buf`.
 * @param  thinking_buf     Out: contents of `<think>…</think>` block.
 * @param  thinking_len     Capacity of `thinking_buf`.
 * @param  tools_buf        Out: JSON array of tool calls (`"[]"` if none).
 * @param  tools_len        Capacity of `tools_buf`.
 * @param  out_has_tools    Out: set to 1 if any tool call was parsed.
 * @return                  0 = ok, -1 = parse error, 1 = a buffer was truncated.
 */
int ion7_chat_parse(
    ion7_chat_templates_t* t,
    const char*            text,
    int                    enable_thinking,
    char*   content_buf,  int32_t content_len,
    char*   thinking_buf, int32_t thinking_len,
    char*   tools_buf,    int32_t tools_len,
    int*    out_has_tools);

/* =========================================================================
 * Reasoning Budget
 * ======================================================================= */

/**
 * Create a reasoning-budget sampler that hard-limits the token count
 * inside `<think>…</think>` blocks for Qwen3/DeepSeek-R1-style models.
 *
 * Insert into a sampler chain via `llama_sampler_chain_add` (called from
 * Lua via FFI). The chain takes ownership; do NOT free manually after
 * adding it to a chain.
 *
 * @param  model     Loaded model (used to tokenize delimiters).
 * @param  n_budget  Max tokens inside the block:
 *                     0  = disable thinking entirely (pre-fill empty block),
 *                    -1  = unlimited (passthrough — useful for dynamic ctrl).
 * @return           Sampler pointer, or NULL on failure.
 */
struct llama_sampler* ion7_reasoning_budget_init(
    const struct llama_model* model,
    int32_t                   n_budget);

/* =========================================================================
 * common_sampler — advanced sampling
 *   DRY, XTC, Mirostat, grammar_lazy, Top-N-sigma, adaptive-P, ...
 * ======================================================================= */

/**
 * Flat parameter struct for `ion7_csampler_init`. Allocate from Lua with
 * `ffi.new("ion7_csampler_params_t")` and set the fields you care about.
 * Unset fields default to 0 / 0.0 — set to llama.cpp defaults explicitly
 * if the zero value is meaningful for that knob (e.g. `top_p = 1.0`).
 */
typedef struct {
    uint32_t seed;              /**< `LLAMA_DEFAULT_SEED` (0xFFFFFFFF) for random.   */
    int32_t  top_k;             /**< Keep top-k tokens; <= 0 disables.               */
    float    top_p;             /**< Nucleus threshold; 1.0 disables.                */
    float    min_p;             /**< Min-p threshold; 0.0 disables.                  */
    float    xtc_probability;   /**< XTC apply probability; 0.0 disables.            */
    float    xtc_threshold;     /**< XTC removal threshold (default 0.1).            */
    float    temp;              /**< Sampling temperature (default 0.8).             */
    float    repeat_penalty;    /**< Repetition penalty; 1.0 disables.               */
    float    freq_penalty;      /**< Frequency penalty; 0.0 disables.                */
    float    pres_penalty;      /**< Presence penalty; 0.0 disables.                 */
    int32_t  repeat_last_n;     /**< Window for repeat penalties; -1 = full ctx.     */
    float    dry_mult;          /**< DRY multiplier; 0.0 disables.                   */
    float    dry_base;          /**< DRY base (default 1.75).                        */
    int32_t  dry_allowed_len;   /**< DRY allowed sequence length (default 2).        */
    int32_t  dry_last_n;        /**< DRY penalty window; -1 = full ctx.              */
    int32_t  mirostat;          /**< 0 = off, 1 = Mirostat v1, 2 = Mirostat v2.      */
    float    mirostat_tau;      /**< Mirostat target entropy (default 5.0).          */
    float    mirostat_eta;      /**< Mirostat learning rate (default 0.1).           */
    float    top_n_sigma;       /**< Top-N-Sigma cutoff; -1.0 disables.              */
    float    adaptive_target;   /**< Adaptive-P target probability; -1.0 disables.   */
    float    adaptive_decay;    /**< Adaptive-P EMA decay (default 0.9).             */
    int32_t  grammar_lazy;      /**< 0 = always apply grammar, 1 = lazy (CRANE).     */
} ion7_csampler_params_t;

/** Opaque handle to a `common_sampler` instance. */
typedef struct ion7_csampler ion7_csampler_t;

/**
 * Create a `common_sampler` from a numeric parameter set, optional GBNF
 * grammar, optional grammar trigger words and optional logit bias map.
 *
 * @param  model            Loaded model (used for vocab + grammar compilation).
 * @param  params           Numeric sampling params; NULL = all defaults.
 * @param  grammar          GBNF grammar string, or NULL.
 * @param  trigger_words    Words that activate `grammar_lazy`; NULL allowed.
 * @param  n_triggers       Length of `trigger_words`.
 * @param  logit_bias_ids   Token IDs for manual logit bias; NULL allowed.
 * @param  logit_bias_val   Bias values matching `logit_bias_ids`.
 * @param  n_logit_bias    Length of both bias arrays.
 * @return                  Opaque handle, or NULL on failure.
 */
ion7_csampler_t* ion7_csampler_init(
    const struct llama_model*     model,
    const ion7_csampler_params_t* params,
    const char*                   grammar,
    const char**                  trigger_words,
    int                           n_triggers,
    const int32_t*                logit_bias_ids,
    const float*                  logit_bias_val,
    int                           n_logit_bias);

/** Free a common_sampler. Safe to call with NULL. */
void ion7_csampler_free(ion7_csampler_t* s);

/**
 * Sample a token from the context's logits at batch index `idx`.
 *
 * @param  grammar_first  1 = apply grammar before other samplers
 *                        (more correct, slightly slower).
 * @return                Sampled token ID, or -1 if `s` is NULL.
 */
int32_t ion7_csampler_sample(ion7_csampler_t* s,
                             struct llama_context* ctx,
                             int idx,
                             int grammar_first);

/** Notify the sampler that `token` was accepted by the chain. */
void ion7_csampler_accept(ion7_csampler_t* s, int32_t token);

/**
 * Combined sample + accept in one bridge call. Preferred hot-path
 * variant: a single FFI boundary crossing per generated token instead
 * of two.
 */
int32_t ion7_csampler_sample_accept(ion7_csampler_t* s,
                                    struct llama_context* ctx,
                                    int idx,
                                    int grammar_first);

/** Reset all sampler state (grammar automaton, DRY history, mirostat). */
void ion7_csampler_reset(ion7_csampler_t* s);

/** Returns the last accepted token, or -1 if none / on error. */
int32_t ion7_csampler_last(const ion7_csampler_t* s);

/** Returns the effective RNG seed used by this sampler instance. */
uint32_t ion7_csampler_get_seed(const ion7_csampler_t* s);

/* =========================================================================
 * Speculative decoding (n-gram / draft model)
 *
 * Constants are 1:1 with `common_speculative_type` in
 * `common/common.h`. Never remap them — breaking changes are intentional
 * when llama.cpp upstream changes.
 * ======================================================================= */

#define ION7_SPEC_NONE          0  /**< No speculative decoding.                 */
#define ION7_SPEC_DRAFT         1  /**< Separate lighter draft model (ctx_dft).  */
#define ION7_SPEC_EAGLE3        2  /**< EAGLE-3 draft heads on target model.     */
#define ION7_SPEC_NGRAM_SIMPLE  3  /**< Simple n-gram from recent context.       */
#define ION7_SPEC_NGRAM_MAP_K   4  /**< N-gram map with prediction statistics.   */
#define ION7_SPEC_NGRAM_MAP_K4V 5  /**< N-gram map with k + 4 m-gram values.     */
#define ION7_SPEC_NGRAM_MOD     6  /**< N-gram with modular prediction.          */
#define ION7_SPEC_NGRAM_CACHE   7  /**< LRU n-gram cache (~ Cacheback paper).    */

/** Opaque handle to a `common_speculative` instance. */
typedef struct ion7_speculative ion7_speculative_t;

/**
 * Create a speculative decoder.
 *
 * @param  ctx_tgt    Main (target) context. Always required.
 * @param  ctx_dft    Draft model context. Required for `ION7_SPEC_DRAFT`,
 *                    NULL otherwise.
 * @param  type       One of the `ION7_SPEC_*` constants above.
 * @param  n_draft    Max draft tokens generated per step (default 16).
 * @param  ngram_min  Minimum n-gram size for n-gram types (default 1).
 * @param  ngram_max  Maximum n-gram size for n-gram types (default 4).
 * @return            Opaque handle, or NULL on failure.
 */
ion7_speculative_t* ion7_speculative_init(
    struct llama_context* ctx_tgt,
    struct llama_context* ctx_dft,
    int type, int n_draft, int ngram_min, int ngram_max);

/** Free a speculative decoder. Safe to call with NULL. */
void ion7_speculative_free(ion7_speculative_t* spec);

/**
 * Notify the decoder of the full current prompt. Call once before the
 * generation loop and again whenever the prompt changes wholesale.
 */
void ion7_speculative_begin(ion7_speculative_t* spec,
                            const int32_t* prompt,
                            int n_prompt);

/**
 * Generate draft tokens.
 *
 * @param  prompt      Full current sequence (prompt + accepted-so-far).
 * @param  n_prompt    Length of `prompt`.
 * @param  last_token  Last token accepted by the target model.
 * @param  out_draft   Buffer receiving draft token IDs.
 * @param  max_draft   Capacity of `out_draft`.
 * @return             Number of tokens written to `out_draft`.
 */
int ion7_speculative_draft(ion7_speculative_t* spec,
                           const int32_t* prompt, int n_prompt,
                           int32_t last_token,
                           int32_t* out_draft, int max_draft);

/** Report how many consecutive draft tokens the target model accepted. */
void ion7_speculative_accept(ion7_speculative_t* spec, int n_accepted);

/** Print acceptance rate and effective speedup to stderr. */
void ion7_speculative_stats(const ion7_speculative_t* spec);

/* =========================================================================
 * JSON Schema → GBNF grammar
 * ======================================================================= */

/**
 * Convert a JSON Schema (draft-07) to a GBNF grammar string.
 * The C++ implementation handles `$ref`, `anyOf`, `allOf`, `oneOf` and
 * format constraints more robustly than the pure-Lua `ion7-grammar`
 * fallback.
 *
 * @param  schema_json  NUL-terminated JSON Schema string.
 * @param  out          Output buffer (NUL-terminated GBNF).
 * @param  out_len      Capacity of `out`.
 * @return              Bytes needed (incl. NUL), or -1 on parse error.
 *                      If the return value > `out_len`, resize and retry.
 */
int ion7_json_schema_to_grammar(const char* schema_json,
                                char* out, size_t out_len);

/* =========================================================================
 * Partial regex matching (streaming stop-string detection)
 * ======================================================================= */

/** Opaque compiled regex handle. */
typedef struct ion7_regex ion7_regex_t;

/** Compile a regex pattern. Returns NULL on invalid pattern. */
ion7_regex_t* ion7_regex_new(const char* pattern);

/** Free a compiled regex. Safe to call with NULL. */
void ion7_regex_free(ion7_regex_t* r);

/**
 * Match the pattern against text. When `partial = 1` also returns 1 for
 * text that is a valid prefix of a future full match — used in streaming
 * to decide whether to keep buffering more tokens before emitting.
 *
 * @return  0 = no match (can stop buffering),
 *          1 = partial prefix match (keep buffering),
 *          2 = full match (trigger stop).
 */
int ion7_regex_search(ion7_regex_t* r, const char* text, size_t len, int partial);

/* =========================================================================
 * JSON utilities (nlohmann/json — already linked via libcommon)
 * String-in / string-out only: no C++ types cross the ABI.
 * ======================================================================= */

/** Validate a JSON string. Returns 1 if valid, 0 otherwise. */
int ion7_json_validate(const char* json_str);

/**
 * Pretty-print a JSON string with 2-space indentation.
 *
 * @param  json_str  Input JSON (compact or already formatted).
 * @param  out       Output buffer; may be NULL for size query.
 * @param  out_len   Capacity of `out`.
 * @return           Bytes needed (incl. NUL), or -1 on parse error.
 */
int ion7_json_format(const char* json_str, char* out, size_t out_len);

/**
 * Merge two JSON objects using RFC 7396 JSON Merge Patch semantics.
 * Keys in `overlay` overwrite matching keys in `base`; null values
 * delete keys.
 *
 * @param  base     Base JSON object string.
 * @param  overlay  Patch JSON object string (applied on top of base).
 * @param  out      Output buffer for the merged JSON.
 * @param  out_len  Capacity of `out`.
 * @return          Bytes needed (incl. NUL), or -1 on parse error.
 */
int ion7_json_merge(const char* base, const char* overlay,
                    char* out, size_t out_len);

/* =========================================================================
 * Training (llama_opt + common_opt_dataset_init)
 * ======================================================================= */

/** Opaque training state handle returned by `ion7_opt_init`. */
typedef struct ion7_opt_state ion7_opt_state_t;

/**
 * Initialise training on a context. Must be called before
 * `ion7_opt_epoch`. The context must have been created with F32 or BF16
 * KV cache (quantised KV is not supported during training).
 *
 * @param  ctx        Inference context.
 * @param  model      Loaded model.
 * @param  optimizer  0 = AdamW (recommended), 1 = SGD.
 * @param  lr         Learning rate (e.g. 1e-4 for LoRA, 1e-5 for full FT).
 * @return            Opaque state handle. Free with `ion7_opt_free`.
 */
ion7_opt_state_t* ion7_opt_init(
    struct llama_context* ctx,
    struct llama_model*   model,
    int                   optimizer,
    float                 lr);

/** Release training state. */
void ion7_opt_free(ion7_opt_state_t* state);

/**
 * Build a training dataset from a pre-tokenised corpus.
 *
 * @param  ctx       Inference context.
 * @param  tokens    Flat array of `llama_token` (entire corpus).
 * @param  n_tokens  Total token count.
 * @param  stride    Tokens between the start of consecutive samples.
 *                   Pass `llama_n_ctx(ctx)` for non-overlapping windows.
 * @return           Dataset handle. Free with `ion7_opt_dataset_free`.
 */
ggml_opt_dataset_t ion7_opt_dataset_create(
    struct llama_context* ctx,
    const llama_token*    tokens,
    int64_t               n_tokens,
    int64_t               stride);

/** Free a dataset. */
void ion7_opt_dataset_free(ggml_opt_dataset_t dataset);

/**
 * Run one training epoch over the dataset.
 *
 * @param  ctx        Inference context (already initialised by `ion7_opt_init`).
 * @param  dataset    Dataset handle from `ion7_opt_dataset_create`.
 * @param  val_split  Fraction of dataset held out for validation
 *                    (`0.0` = no validation).
 * @return            Average training loss, or `-1.0` on error.
 */
float ion7_opt_epoch(
    struct llama_context* ctx,
    ggml_opt_dataset_t    dataset,
    float                 val_split);

#ifdef __cplusplus
}
#endif

#endif /* ION7_BRIDGE_H */
