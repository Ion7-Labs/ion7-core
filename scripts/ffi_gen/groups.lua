--- @module ffi_gen.groups
--- @author  ion7 / generate_ffi.lua
---
--- Configuration tables describing how llama.cpp / ggml / gguf functions are
--- bucketed into per-domain Lua FFI files.
---
--- Each `*_GROUPS` value is an ordered array of `{ group_name, prefixes[] }`.
--- A function name is matched against prefixes in declaration order; the FIRST
--- matching prefix wins. Anything that matches no prefix lands in `misc.lua`.
---
--- ORDER MATTERS:
---  - More specific groups must come BEFORE catch-all ones. For instance the
---    backend-specific groups (cuda, metal, vulkan...) must precede the generic
---    `backend` group, otherwise `ggml_backend_cuda_*` would all be swallowed
---    by the catch-all and would never reach `cuda.lua`.
---
--- Adding a new group:
---  1. Insert `{ "name", { "prefix_", ... } }` BEFORE the relevant catch-all.
---  2. Re-run `luajit scripts/generate_ffi.lua` and verify with the coverage
---     report that no expected functions silently land in `misc`.

local M = {}

--- Function-name buckets for `llama_*` symbols → `ffi/llama/<group>.lua`.
--- @type table<integer, { [1]: string, [2]: string[] }>
M.LLAMA_GROUPS = {
  { "model",   { "llama_model_", "llama_load_model_", "llama_free_model" } },
  { "context", { "llama_context_", "llama_new_context_", "llama_free", "llama_n_ctx",
                 "llama_n_batch", "llama_n_ubatch", "llama_n_seq_max",
                 "llama_decode", "llama_encode", "llama_set_", "llama_synchronize" } },
  { "batch",   { "llama_batch_" } },
  { "sampler", { "llama_sampler_" } },
  { "vocab",   { "llama_vocab_", "llama_token_", "llama_tokenize", "llama_detokenize" } },
  { "kv_cache",{ "llama_kv_cache_", "llama_memory_" } },
  { "adapter", { "llama_adapter_", "llama_set_adapter_lora" } },
  { "state",   { "llama_state_" } },
  { "chat",    { "llama_chat_" } },
  { "perf",    { "llama_perf_" } },
  { "opt",     { "llama_opt_" } },
  { "logits",  { "llama_get_logits", "llama_get_embeddings" } },
  { "backend", { "llama_backend_", "llama_numa_", "llama_supports_",
                 "llama_max_devices", "llama_max_parallel_sequences",
                 "llama_log_set", "llama_print_system_info",
                 "llama_time_us", "llama_split_path", "llama_split_prefix" } },
}

--- Function-name buckets for `ggml_*` symbols → `ffi/ggml/<group>.lua`.
---
--- The backend-specific groups (cpu/blas/vulkan/cuda/metal/opencl/sycl/rpc/
--- webgpu) come BEFORE the catch-all `backend` group on purpose: see the
--- ORDER MATTERS note above.
--- @type table<integer, { [1]: string, [2]: string[] }>
M.GGML_GROUPS = {
  { "tensor",   { "ggml_new_tensor", "ggml_dup_tensor", "ggml_view_", "ggml_reshape_",
                  "ggml_permute", "ggml_transpose", "ggml_cont", "ggml_set_",
                  "ggml_get_", "ggml_nbytes", "ggml_nrows", "ggml_nelements",
                  "ggml_n_dims", "ggml_blck_size", "ggml_element_size",
                  "ggml_type_size", "ggml_row_size", "ggml_is_" } },
  { "ops",      { "ggml_add", "ggml_sub", "ggml_mul", "ggml_div", "ggml_sqr", "ggml_sqrt",
                  "ggml_log", "ggml_sum", "ggml_mean", "ggml_norm", "ggml_rms_norm",
                  "ggml_rope", "ggml_alibi", "ggml_clamp", "ggml_soft_max", "ggml_silu",
                  "ggml_relu", "ggml_gelu", "ggml_mul_mat", "ggml_out_prod",
                  "ggml_concat", "ggml_pad", "ggml_repeat", "ggml_argmax", "ggml_top_k" } },
  { "context",  { "ggml_init", "ggml_free", "ggml_used_mem", "ggml_get_mem_buffer",
                  "ggml_get_mem_size", "ggml_set_no_alloc" } },
  { "graph",    { "ggml_new_graph", "ggml_build_forward_", "ggml_graph_",
                  "ggml_visit_parents" } },
  { "quantize", { "ggml_quantize_", "ggml_dequantize_" } },
  { "alloc",    { "ggml_gallocr_", "ggml_tallocr_" } },
  { "opt",      { "ggml_opt_" } },
  -- Backend-specific buckets MUST precede the catch-all `backend` below.
  { "cpu",      { "ggml_cpu_", "ggml_backend_cpu_" } },
  { "blas",     { "ggml_backend_blas_" } },
  { "vulkan",   { "ggml_vk_", "ggml_backend_vk_" } },
  { "cuda",     { "ggml_cuda_", "ggml_backend_cuda_" } },
  { "metal",    { "ggml_metal_", "ggml_backend_metal_" } },
  { "opencl",   { "ggml_opencl_", "ggml_backend_opencl_" } },
  { "sycl",     { "ggml_sycl_", "ggml_backend_sycl_" } },
  { "rpc",      { "ggml_rpc_", "ggml_backend_rpc_" } },
  { "webgpu",   { "ggml_webgpu_", "ggml_backend_webgpu_" } },
  -- Catch-all : matches whatever ggml_backend_* did not land elsewhere.
  { "backend",  { "ggml_backend_" } },
  { "misc",     { "ggml_print_", "ggml_format_", "ggml_op_", "ggml_unary_op_",
                  "ggml_status_to_string", "ggml_time_", "ggml_fopen",
                  "ggml_log_set", "ggml_abort" } },
}

--- Function-name buckets for `gguf_*` symbols → `ffi/gguf/<group>.lua`.
--- @type table<integer, { [1]: string, [2]: string[] }>
M.GGUF_GROUPS = {
  { "gguf", { "gguf_" } },
}

--- Macro names that should NEVER be emitted as declarations. These are
--- visibility / calling-convention annotations (defined-out by the clang
--- invocation in `ffi_gen.clang`).
--- @type table<string, true>
M.IGNORED_MACROS = {
  ["LLAMA_API"]      = true,
  ["LLAMA_DEPRECATED"] = true,
  ["GGML_API"]       = true,
  ["GGML_CALL"]      = true,
  ["GGML_DEPRECATED"] = true,
  ["__attribute__"]  = true,
}

--- Default header set parsed when no `--include-*` flag is passed. These cover
--- ~95% of ion7 use cases (CPU, BLAS, Vulkan, optimisation, GGUF on-disk).
---
--- Paths are relative to `<vendor_root>/llama.cpp/`.
--- @type string[]
M.DEFAULT_HEADERS = {
  "include/llama.h",
  "ggml/include/ggml.h",
  "ggml/include/ggml-alloc.h",
  "ggml/include/ggml-backend.h",
  "ggml/include/ggml-cpu.h",
  "ggml/include/ggml-blas.h",
  "ggml/include/ggml-opt.h",
  "ggml/include/gguf.h",
}

--- Optional headers, opt-in via dedicated CLI flags.
---
--- The key is the flag basename (e.g. `vulkan` → `--include-vulkan`), the
--- value is the relative header path. `vulkan` is included by default; the
--- rest must be requested explicitly because they require backend toolchains
--- that may not exist on the developer's machine.
--- @type table<string, string>
M.OPTIONAL_HEADERS = {
  vulkan  = "ggml/include/ggml-vulkan.h",
  cuda    = "ggml/include/ggml-cuda.h",
  metal   = "ggml/include/ggml-metal.h",
  opencl  = "ggml/include/ggml-opencl.h",
  sycl    = "ggml/include/ggml-sycl.h",
  rpc     = "ggml/include/ggml-rpc.h",
  webgpu  = "ggml/include/ggml-webgpu.h",
}

--- Preprocessor defines passed to clang to neutralise visibility / calling
--- convention macros that would otherwise pollute the AST.
--- @type string[]
M.CLANG_DEFINES = {
  "-DLLAMA_API=",
  "-DGGML_API=",
  "-DGGML_CALL=",
  "-DGGML_DEPRECATED(...)=",
}

--- Include-path subdirectories under `<vendor_root>/llama.cpp/` passed to
--- clang via `-I`. Order matters: the first match wins, so the most specific
--- comes first.
--- @type string[]
M.INCLUDE_SUBDIRS = {
  "include",
  "ggml/include",
  "common",
}

return M
