/*
 * bridge_chat.cpp - Chat templates, output parsing, reasoning budget.
 *
 * Copyright (C) 2026 Ion7 Project Contributors
 * SPDX-License-Identifier: MIT
 *
 * Wraps three closely-related libcommon components:
 *   - common_chat_templates       : Jinja2 chat-template rendering
 *   - common_chat_parse           : structured parsing of model output
 *                                   (content + reasoning + tool calls)
 *   - common_reasoning_budget     : sampler that caps tokens inside
 *                                   <think>…</think> blocks
 *
 * The C ABI exposed in `ion7_bridge.h` hides every C++ type. All
 * incoming and outgoing strings are plain `char*` buffers; tool calls
 * are serialised to a JSON string via nlohmann/json so that the caller
 * never has to walk a C++ collection.
 */

#include "ion7_bridge.h"
#include "bridge_internal.hpp"

#include "chat.h"
#include "common.h"
#include "reasoning-budget.h"

#include "nlohmann/json.hpp"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

/* =========================================================================
 * Chat templates (Jinja2 native, enable_thinking support)
 * =========================================================================
 *
 * `ion7_chat_templates_priv` wraps the libcommon templates handle AND a
 * memo of the most recent `common_chat_params` returned by
 * `common_chat_templates_apply`. Storing that memo lets `ion7_chat_parse`
 * later call `common_chat_parse` with the correct PEG parser instead of
 * the pure-content default — which is what lets us extract tool calls
 * from Mistral / Qwen / Hermes templates without the caller having to
 * know which family the model belongs to.
 *
 * The memo carries everything `common_chat_parser_params` needs :
 *   - `format`            — selected `common_chat_format` enum value.
 *   - `generation_prompt` — pinned by some templates that prepend a tag.
 *   - `parser`            — serialised PEG arena (a flat string).
 *
 * Each `_apply` call overwrites the memo. That is fine for the single
 * "apply → parse" cycle ion7-llm runs per turn ; if a future caller
 * needs to apply once and parse many times across different inputs, a
 * dedicated `ion7_chat_session_t` opaque could be added later without
 * breaking this API.
 */
struct ion7_chat_templates_priv {
    common_chat_templates * tmpls;       /* owned */
    common_chat_params      last_apply;  /* refreshed on every _apply */
    bool                    has_apply;
};

extern "C"
ion7_chat_templates_t* ion7_chat_templates_init(
    const struct llama_model* model,
    const char*               tmpl_override)
{
    auto raw = common_chat_templates_init(
        model,
        tmpl_override ? tmpl_override : "",
        "",   /* bos_token: model default */
        "");  /* eos_token: model default */
    if (!raw) return nullptr;

    auto * priv = new ion7_chat_templates_priv();
    priv->tmpls     = raw.release();
    priv->has_apply = false;
    return reinterpret_cast<ion7_chat_templates_t*>(priv);
}

extern "C"
void ion7_chat_templates_free(ion7_chat_templates_t* t)
{
    if (!t) return;
    auto * priv = reinterpret_cast<ion7_chat_templates_priv*>(t);
    if (priv->tmpls) common_chat_templates_free(priv->tmpls);
    delete priv;
}

extern "C"
int ion7_chat_templates_support_thinking(const ion7_chat_templates_t* t)
{
    if (!t) return 0;
    auto * priv = reinterpret_cast<const ion7_chat_templates_priv*>(t);
    return common_chat_templates_support_enable_thinking(priv->tmpls) ? 1 : 0;
}

extern "C"
int32_t ion7_chat_templates_apply(
    ion7_chat_templates_t* t,
    const char**           roles,
    const char**           contents,
    size_t                 n_msgs,
    int                    add_ass,
    int                    enable_thinking,
    char*                  buf,
    int32_t                buf_len)
{
    if (!t) return -1;
    auto * priv = reinterpret_cast<ion7_chat_templates_priv*>(t);

    common_chat_templates_inputs inputs;
    inputs.add_generation_prompt = (bool)add_ass;
    inputs.use_jinja             = true;

    /* enable_thinking tri-state: -1 leaves the libcommon default in place. */
    if      (enable_thinking == 0) inputs.enable_thinking = false;
    else if (enable_thinking == 1) inputs.enable_thinking = true;

    inputs.messages.reserve(n_msgs);
    for (size_t i = 0; i < n_msgs; i++) {
        common_chat_msg msg;
        msg.role    = roles    && roles[i]    ? roles[i]    : "";
        msg.content = contents && contents[i] ? contents[i] : "";
        inputs.messages.push_back(std::move(msg));
    }

    common_chat_params result;
    try {
        result = common_chat_templates_apply(priv->tmpls, inputs);
    } catch (const std::exception&) {
        return -2;
    } catch (...) {
        return -2;
    }

    /* Guard against prompts ≥ 2 GB: size_t → int32_t would silently
     * wrap and a downstream memcpy would receive a huge length. */
    if (result.prompt.size() >= (size_t)INT32_MAX) return -3;

    int32_t needed = (int32_t)result.prompt.size() + 1;
    if (buf && buf_len >= needed) {
        std::memcpy(buf, result.prompt.c_str(), (size_t)needed);
    }

    /* Memo the params for the next _chat_parse call. We move the result
     * to avoid copying the (potentially large) serialised PEG parser
     * string and the prompt itself (the prompt is already in `buf`).
     * `has_apply` flips on the first successful render. */
    priv->last_apply = std::move(result);
    priv->has_apply  = true;

    return needed;
}

/* =========================================================================
 * Chat output parsing (tool calls + reasoning extraction)
 * ======================================================================= */

extern "C"
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
    auto * priv = reinterpret_cast<ion7_chat_templates_priv*>(t);

    common_chat_parser_params pparams;
    pparams.parse_tool_calls = true;
    pparams.reasoning_format = (enable_thinking != 0)
        ? COMMON_REASONING_FORMAT_AUTO
        : COMMON_REASONING_FORMAT_NONE;

    /* Promote the memoised render's metadata into the parser params.
     * Without this, libcommon falls back to the pure-content PEG parser
     * and tool calls go undetected for templates whose tool-call
     * envelope is non-default (Mistral / Qwen / Hermes / …). */
    if (priv->has_apply) {
        pparams.format            = priv->last_apply.format;
        pparams.generation_prompt = priv->last_apply.generation_prompt;
        if (!priv->last_apply.parser.empty()) {
            try {
                pparams.parser.load(priv->last_apply.parser);
            } catch (...) {
                /* Parser deserialisation failed — fall back to the
                 * default content-only behaviour rather than aborting
                 * the whole parse. */
                pparams.parser = {};
            }
        }
    }

    common_chat_msg msg;
    try {
        msg = common_chat_parse(text, false, pparams);
    } catch (...) {
        return -1;
    }

    int truncated = 0;
    ion7_safe_copy(msg.content,           content_buf,  content_len,  &truncated);
    ion7_safe_copy(msg.reasoning_content, thinking_buf, thinking_len, &truncated);

    /* Serialise tool_calls as a JSON array.
     *
     * `tc.arguments` already holds JSON text, so we parse it back into a
     * nlohmann tree to embed it as a proper sub-object rather than as a
     * quoted string. Malformed argument JSON is preserved verbatim as a
     * string so the caller can still inspect what the model emitted. */
    if (msg.tool_calls.empty()) {
        if (tools_buf && tools_len >= 3) {
            std::memcpy(tools_buf, "[]", 3);
        }
        if (out_has_tools) *out_has_tools = 0;
    } else {
        try {
            nlohmann::json arr = nlohmann::json::array();
            for (const auto& tc : msg.tool_calls) {
                nlohmann::json obj;
                obj["name"] = tc.name;
                obj["id"]   = tc.id;
                try {
                    obj["arguments"] = nlohmann::json::parse(tc.arguments);
                } catch (...) {
                    obj["arguments"] = tc.arguments;
                }
                arr.push_back(std::move(obj));
            }
            std::string serialised = arr.dump();
            ion7_safe_copy(serialised, tools_buf, tools_len, &truncated);
        } catch (...) {
            if (tools_buf && tools_len >= 3) {
                std::memcpy(tools_buf, "[]", 3);
            }
        }
        if (out_has_tools) *out_has_tools = 1;
    }
    return truncated ? 1 : 0;
}

/* =========================================================================
 * Reasoning budget sampler
 * ======================================================================= */

/**
 * Tokenise a literal string with the model's vocab. Returns an empty
 * vector on failure (error reporting is intentionally silent — the
 * caller treats an empty delimiter as "feature disabled" gracefully).
 */
static std::vector<llama_token> tokenize_literal(const llama_vocab* vocab,
                                                 const std::string& s)
{
    std::vector<llama_token> toks(s.size() + 8);
    int n = llama_tokenize(vocab,
                           s.c_str(),
                           (int32_t)s.size(),
                           toks.data(),
                           (int32_t)toks.size(),
                           /*add_special=*/false,
                           /*parse_special=*/true);
    if (n < 0) return {};
    toks.resize(n);
    return toks;
}

extern "C"
struct llama_sampler* ion7_reasoning_budget_init(const struct llama_model* model,
                                                 int32_t                   n_budget)
{
    const llama_vocab* vocab = llama_model_get_vocab(model);

    /* Standard delimiters used by Qwen3, DeepSeek-R1 and other
     * reasoning models. The "forced" sequence is what the sampler
     * inserts when the budget is exhausted; here it is identical to
     * the closing delimiter so the model resumes normal output cleanly. */
    auto start_toks   = tokenize_literal(vocab, "<think>\n");
    auto end_toks     = tokenize_literal(vocab, "\n</think>\n");
    auto forced_toks  = end_toks;
    auto prefill_toks = tokenize_literal(vocab, "<think>\n\n</think>\n");

    return common_reasoning_budget_init(vocab,
                                        start_toks,
                                        end_toks,
                                        forced_toks,
                                        n_budget,
                                        prefill_toks);
}
