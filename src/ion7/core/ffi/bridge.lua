--- @module ion7.core.ffi.bridge
--- @author  ion7 / Ion7 Project Contributors
---
--- Hand-written FFI binding for `bridge/ion7_bridge.so` — the libcommon
--- C++ adapter layer.
---
--- Why this file is hand-written rather than emitted by `generate_ffi.lua` :
---
---   - `ion7_bridge.h` is a stable HUMAN-CURATED API designed to never
---     change across llama.cpp bumps, so re-running a generator on every
---     bump would be pointless churn.
---   - The bridge intentionally pins a small surface (~34 functions) to
---     hide libcommon's C++ ABI ; running the generator on it would force
---     us to also feed in `chat.h`, `sampling.h`, etc. and fight the C++
---     mangling — exactly what the bridge was created to avoid.
---
--- Loader strategy :
---   The shared library is searched in this order, allowing both dev and
---   container layouts to "just work" without configuration :
---     1. `ION7_BRIDGE_PATH` environment variable (explicit override).
---     2. `bridge/ion7_bridge.so` relative to CWD (typical dev layout).
---     3. `ion7_bridge.so` on the system loader path (production install).
---
--- Returns the `ffi.load` handle so callers can do :
---   local bridge = require "ion7.core.ffi.bridge"
---   bridge.ion7_csampler_init(...)

local ffi = require "ffi"
require "ion7.core.ffi.types" -- needed for llama_*, ggml_* type names

ffi.cdef [[
  /* ── Version + VRAM auto-fit ──────────────────────────────────────── */
  const char* ion7_bridge_version(void);

  int ion7_params_fit(const char* path,
                      int32_t* n_gpu_layers,
                      uint32_t* n_ctx,
                      uint32_t  n_ctx_min);

  /* ── Chat templates (Jinja2) ──────────────────────────────────────── */
  typedef struct ion7_chat_templates_priv ion7_chat_templates_t;

  ion7_chat_templates_t* ion7_chat_templates_init(
      const struct llama_model* model,
      const char*               tmpl_override);

  void ion7_chat_templates_free(ion7_chat_templates_t* t);

  int ion7_chat_templates_support_thinking(const ion7_chat_templates_t* t);

  int32_t ion7_chat_templates_apply(
      ion7_chat_templates_t* t,
      const char**           roles,
      const char**           contents,
      size_t                 n_msgs,
      int                    add_ass,
      int                    enable_thinking,
      char*                  buf,
      int32_t                buf_len);

  /* ── Chat output parsing ──────────────────────────────────────────── */
  int ion7_chat_parse(
      ion7_chat_templates_t* t,
      const char*            text,
      int                    enable_thinking,
      char*   content_buf,  int32_t content_len,
      char*   thinking_buf, int32_t thinking_len,
      char*   tools_buf,    int32_t tools_len,
      int*    out_has_tools);

  /* ── Reasoning budget ─────────────────────────────────────────────── */
  struct llama_sampler* ion7_reasoning_budget_init(
      const struct llama_model* model,
      int32_t                   n_budget);

  /* ── common_sampler (DRY/XTC/Mirostat/grammar_lazy/...) ───────────── */
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
      float    top_n_sigma;
      float    adaptive_target;
      float    adaptive_decay;
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

  void    ion7_csampler_free(ion7_csampler_t* s);

  int32_t ion7_csampler_sample(ion7_csampler_t* s,
                               struct llama_context* ctx,
                               int idx, int grammar_first);

  void    ion7_csampler_accept(ion7_csampler_t* s, int32_t token);

  int32_t ion7_csampler_sample_accept(ion7_csampler_t* s,
                                       struct llama_context* ctx,
                                       int idx, int grammar_first);

  void    ion7_csampler_reset(ion7_csampler_t* s);
  int32_t ion7_csampler_last(const ion7_csampler_t* s);
  uint32_t ion7_csampler_get_seed(const ion7_csampler_t* s);

  /* ── Speculative decoding ─────────────────────────────────────────── */
  typedef struct ion7_speculative ion7_speculative_t;

  ion7_speculative_t* ion7_speculative_init(
      struct llama_context* ctx_tgt,
      struct llama_context* ctx_dft,
      int type, int n_draft, int ngram_min, int ngram_max);

  void ion7_speculative_free(ion7_speculative_t* spec);

  void ion7_speculative_begin(ion7_speculative_t* spec,
                              const int32_t* prompt, int n_prompt);

  int  ion7_speculative_draft(ion7_speculative_t* spec,
                              const int32_t* prompt, int n_prompt,
                              int32_t last_token,
                              int32_t* out_draft, int max_draft);

  void ion7_speculative_accept(ion7_speculative_t* spec, int n_accepted);
  void ion7_speculative_stats (const ion7_speculative_t* spec);

  /* ── JSON Schema → GBNF ───────────────────────────────────────────── */
  int ion7_json_schema_to_grammar(const char* schema_json,
                                  char* out, size_t out_len);

  /* ── Partial regex (streaming stop-string) ────────────────────────── */
  typedef struct ion7_regex ion7_regex_t;

  ion7_regex_t* ion7_regex_new(const char* pattern);
  void          ion7_regex_free(ion7_regex_t* r);
  int           ion7_regex_search(ion7_regex_t* r, const char* text,
                                  size_t len, int partial);

  /* ── JSON utilities (nlohmann-backed) ─────────────────────────────── */
  int ion7_json_validate(const char* json_str);
  int ion7_json_format  (const char* json_str, char* out, size_t out_len);
  int ion7_json_merge   (const char* base,    const char* overlay,
                          char* out, size_t out_len);

  /* ── Training (llama_opt + common_opt_dataset_init) ───────────────── */
  typedef struct ion7_opt_state ion7_opt_state_t;

  ion7_opt_state_t* ion7_opt_init(
      struct llama_context* ctx,
      struct llama_model*   model,
      int                   optimizer,
      float                 lr);

  void               ion7_opt_free(ion7_opt_state_t* state);

  ggml_opt_dataset_t ion7_opt_dataset_create(
      struct llama_context* ctx,
      const llama_token*    tokens,
      int64_t               n_tokens,
      int64_t               stride);

  void  ion7_opt_dataset_free(ggml_opt_dataset_t dataset);

  float ion7_opt_epoch(struct llama_context* ctx,
                       ggml_opt_dataset_t    dataset,
                       float                 val_split);
]]

-- ── Shared-library loader ─────────────────────────────────────────────────

--- Resolve the directory of THIS file so candidate `_libs/ion7_bridge.*`
--- paths can be tried before falling back to the dev-layout / system
--- loader paths.
local function _module_dir()
    local info = debug.getinfo(1, "S")
    local src  = info and info.source or ""
    if src:sub(1, 1) ~= "@" then return nil end
    return (src:sub(2):gsub("/[^/]*$", ""))
end

--- Build the dense candidate list for `ffi.load`. The env var slot is
--- only appended when set ; this avoids putting `nil` in the table,
--- which would make `#` and `ipairs` behave unpredictably under Lua
--- 5.1 / LuaJIT.
---
--- @return string[]
local function bridge_candidates()
    local c = {}
    local env = os.getenv("ION7_BRIDGE_PATH")
    if env and env ~= "" then c[#c + 1] = env end
    -- Sibling `_libs/` populated by the rockspec install step.
    local mdir = _module_dir()
    if mdir then
        c[#c + 1] = mdir .. "/../_libs/ion7_bridge.so"
        c[#c + 1] = mdir .. "/../_libs/ion7_bridge.dylib"
        c[#c + 1] = mdir .. "/../_libs/ion7_bridge.dll"
    end
    c[#c + 1] = "bridge/ion7_bridge.so"
    c[#c + 1] = "bridge/ion7_bridge.dll"
    c[#c + 1] = "ion7_bridge.so"
    c[#c + 1] = "ion7_bridge.dll"
    return c
end

--- Try every candidate path in order ; return the first that loads.
--- @return cdata `ffi.load` handle.
--- @raise   When no candidate can be opened.
local function load_first()
    local candidates = bridge_candidates()
    for _, path in ipairs(candidates) do
        local ok, lib = pcall(ffi.load, path)
        if ok then return lib end
    end
    error(
        string.format(
            "[ion7-core] ion7_bridge.so not found. Candidates : %s\n" ..
                "Build it from `bridge/` with `make LIB_DIR=...` or set " ..
                    "ION7_BRIDGE_PATH to the .so / .dll location.",
            table.concat(candidates, " | ")
        ),
        0
    )
end

return load_first()
