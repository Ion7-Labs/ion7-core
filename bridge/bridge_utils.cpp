/* bridge_utils.cpp — standalone utilities (warmup, UTF-8, JSON schema, regex, cvec, NUMA, CPU caps, log routing, base64, JSON) */
/*
 * Copyright (C) 2026 Ion7 Project Contributors
 * SPDX-License-Identifier: MIT
 */

#include "ion7_bridge.h"
#include "llama.h"
#include "ggml-cpu.h"
#include "bridge_internal.hpp"

/* libcommon utilities */
#include "regex-partial.h"
#include "json-schema-to-grammar.h"
#include "base64.hpp"

/* nlohmann/json — full implementation needed for ordered_json::parse() */
#include "nlohmann/json.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <string>
#include <vector>

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

int ion7_cvec_apply(struct llama_context* ctx, const float* data, size_t len, int32_t n_embd, int32_t il_start, int32_t il_end)
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
    llama_log_set(ion7_log_dispatch, nullptr);
}

void ion7_log_set_timestamps(int enable)
{
    g_log_timestamps = enable ? 1 : 0;
    llama_log_set(ion7_log_dispatch, nullptr);
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

int ion7_json_merge(const char* base, const char* overlay, char* out, size_t out_len)
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
