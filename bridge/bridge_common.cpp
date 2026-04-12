/* bridge_common.cpp - libcommon C++ layer (chat templates, sampling, speculative, training) */
/*
 * Copyright (C) 2026 Ion7 Project Contributors
 * SPDX-License-Identifier: MIT
 */

#include "ion7_bridge.h"

/* libcommon - C++ layer */
#include "chat.h"
#include "common.h"
#include "reasoning-budget.h"
#include "sampling.h"
#include "speculative.h"

/* nlohmann/json - for safe tool-call serialisation in ion7_chat_parse */
#include "nlohmann/json.hpp"

#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

/* =========================================================================
 * ── Chat Templates (Jinja2 native, enable_thinking support) ───────────── */

ion7_chat_templates_t* ion7_chat_templates_init(const struct llama_model* model, const char* tmpl_override)
{
    auto ptr = common_chat_templates_init(model,
        tmpl_override ? tmpl_override : "",
        "",   // bos_token: model default
        "");  // eos_token: model default
    return (ion7_chat_templates_t*)ptr.release();
}

void ion7_chat_templates_free(ion7_chat_templates_t* t)
{
    if (t) common_chat_templates_free((common_chat_templates*)t);
}

int ion7_chat_templates_support_thinking(const ion7_chat_templates_t* t)
{
    if (!t) return 0;
    return common_chat_templates_support_enable_thinking((const common_chat_templates*)t) ? 1 : 0;
}

/*
 * ion7_chat_templates_apply - apply a Jinja2 template with advanced options.
 *
 * roles[]    : array of n_msgs role strings ("system", "user", "assistant")
 * contents[] : array of n_msgs content strings
 * add_ass    : 1 = append assistant generation prefix
 * enable_thinking : -1 = model default, 0 = force off, 1 = force on
 *
 * Returns total bytes needed. If > buf_len, resize buf and call again.
 */
int32_t ion7_chat_templates_apply(ion7_chat_templates_t* t, const char** roles, const char** contents, size_t n_msgs, int add_ass, int enable_thinking, char* buf, int32_t buf_len)
{
    if (!t) return -1;

    common_chat_templates_inputs inputs;
    inputs.add_generation_prompt = (bool)add_ass;
    inputs.use_jinja             = true;

    if (enable_thinking == 0)       inputs.enable_thinking = false;
    else if (enable_thinking == 1)  inputs.enable_thinking = true;
    /* -1 = leave at default (true) */

    inputs.messages.reserve(n_msgs);
    for (size_t i = 0; i < n_msgs; i++) {
        common_chat_msg msg;
        msg.role    = roles[i]    ? roles[i]    : "";
        msg.content = contents[i] ? contents[i] : "";
        inputs.messages.push_back(std::move(msg));
    }

    common_chat_params result;
    try {
        result = common_chat_templates_apply((const common_chat_templates*)t, inputs);
    } catch (const std::exception&) {
        return -2;  // template application failed (e.g. system-only message)
    } catch (...) {
        return -2;
    }

    /* Guard against prompts > 2GB: size_t → int32_t would overflow and
     * make 'needed' negative, causing memcpy to receive a huge size_t value. */
    if (result.prompt.size() >= (size_t)INT32_MAX) return -3;
    int32_t needed = (int32_t)result.prompt.size() + 1;
    if (buf && buf_len >= needed) {
        memcpy(buf, result.prompt.c_str(), (size_t)needed);
    }
    return needed;
}

/* =========================================================================
 * ── Reasoning Budget ──────────────────────────────────────────────────── */

/*
 * ion7_reasoning_budget_init - create a sampler that hard-limits the number
 * of tokens generated inside a <think> block.
 *
 * model   : used to tokenize the special tokens
 * n_budget: max tokens inside the block (0 = disable thinking entirely)
 *
 * Returns a llama_sampler* ready to be inserted into a sampler chain.
 * The chain takes ownership; do not free manually after adding.
 */
struct llama_sampler* ion7_reasoning_budget_init(const struct llama_model* model, int32_t n_budget)
{
    const llama_vocab* vocab = llama_model_get_vocab(model);

    /* Thinking block delimiters:
     * start   : "<think>\n"
     * end     : "\n</think>\n"
     * prefill : "<think>\n\n</think>\n"  (enable_thinking=false) */
    auto tokenize_str = [&](const std::string& s) -> std::vector<llama_token> {
        std::vector<llama_token> toks(s.size() + 8);
        int n = llama_tokenize(vocab, s.c_str(), (int32_t)s.size(), toks.data(), (int32_t)toks.size(), false, true);
        if (n < 0) return {};
        toks.resize(n);
        return toks;
    };

    auto start_toks   = tokenize_str("<think>\n");
    auto end_toks     = tokenize_str("\n</think>\n");
    auto forced_toks  = end_toks;   /* same delimiter as end_toks */
    auto prefill_toks = tokenize_str("<think>\n\n</think>\n");

    return common_reasoning_budget_init(vocab, start_toks, end_toks, forced_toks, n_budget, prefill_toks);
}

/* =========================================================================
 * ── common_sampler (DRY, XTC, grammar_lazy, mirostat, logit bias) ────────
 * ======================================================================= */

/* common_sampler with its owning parameters */
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

    auto* w = new (std::nothrow) ion7_csampler;
    if (!w) return nullptr;
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

int32_t ion7_csampler_sample(ion7_csampler_t* s, struct llama_context* ctx, int idx, int grammar_first)
{
    if (!s) return -1;
    return common_sampler_sample(s->smpl, ctx, idx, (bool)grammar_first);
}

void ion7_csampler_accept(ion7_csampler_t* s, int32_t token)
{
    if (s) common_sampler_accept(s->smpl, token, true);
}

/* ion7_csampler_sample_accept - sample + accept in one call */
int32_t ion7_csampler_sample_accept(ion7_csampler_t* s, struct llama_context* ctx, int idx, int grammar_first)
{
    if (!s) return -1;
    llama_token tok = common_sampler_sample(s->smpl, ctx, idx, (bool)grammar_first);
    common_sampler_accept(s->smpl, tok, true);
    return tok;
}

void ion7_csampler_reset(ion7_csampler_t* s)
{
    if (s) common_sampler_reset(s->smpl);
}

int32_t ion7_csampler_last(const ion7_csampler_t* s)
{
    if (!s) return -1;
    /* common_sampler_last throws if the ring buffer is empty; returns -1 */
    try {
        return common_sampler_last(s->smpl);
    } catch (...) {
        return -1;
    }
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
    common_speculative*       spec;
    common_params_speculative params;
    llama_tokens              toks_buf; /* prompt buffer, shared between begin() and draft() */
};

ion7_speculative_t* ion7_speculative_init(
    struct llama_context* ctx_tgt,
    struct llama_context* ctx_dft,
    int type, int n_draft, int ngram_min, int ngram_max)
{
    auto* w = new (std::nothrow) ion7_speculative;
    if (!w) return nullptr;
    w->params.type      = (common_speculative_type)type;
    w->params.n_max     = (n_draft > 0) ? n_draft : 16;
    w->params.n_min     = 0;
    if (ngram_min > 0) w->params.ngram_size_n = (uint16_t)ngram_min;
    if (ngram_max > 0) w->params.ngram_size_m = (uint16_t)ngram_max;
    /* draft model required only for ION7_SPEC_DRAFT */
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

void ion7_speculative_begin(ion7_speculative_t* spec, const int32_t* prompt, int n_prompt)
{
    if (!spec || !prompt) return;
    spec->toks_buf.assign(prompt, prompt + n_prompt);
    common_speculative_begin(spec->spec, spec->toks_buf);
}

int ion7_speculative_draft(ion7_speculative_t* spec, const int32_t* prompt, int n_prompt, int32_t last_token, int32_t* out_draft, int max_draft)
{
    if (!spec || !prompt || !out_draft) return 0;
    spec->toks_buf.assign(prompt, prompt + n_prompt);
    llama_tokens draft = common_speculative_draft(
        spec->spec, spec->params, spec->toks_buf, last_token);
    int n = (int)std::min((size_t)max_draft, draft.size());
    memcpy(out_draft, draft.data(), (size_t)n * sizeof(int32_t));
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

int ion7_chat_parse(ion7_chat_templates_t* t, const char* text, int enable_thinking, char* content_buf, int32_t content_len, char* thinking_buf, int32_t thinking_len, char* tools_buf, int32_t tools_len, int* out_has_tools)
{
    if (!t || !text) return -1;

    common_chat_parser_params pparams;
    pparams.parse_tool_calls  = true;
    pparams.reasoning_format  = (enable_thinking != 0) ? COMMON_REASONING_FORMAT_AUTO : COMMON_REASONING_FORMAT_NONE;

    common_chat_msg msg = common_chat_parse(text, false, pparams);

    int truncated = 0;
    safe_copy(msg.content, content_buf, content_len, &truncated);
    safe_copy(msg.reasoning_content, thinking_buf, thinking_len, &truncated);

    /* Serialise tool_calls as a JSON array using nlohmann for correct escaping.
     * tc.arguments is already a JSON value from the parser — parse it back so
     * nlohmann embeds it as a proper sub-object rather than a quoted string. */
    if (msg.tool_calls.empty()) {
        if (tools_buf && tools_len >= 3) { memcpy(tools_buf, "[]", 3); }
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
                    /* Malformed arguments: embed as raw string rather than drop. */
                    obj["arguments"] = tc.arguments;
                }
                arr.push_back(std::move(obj));
            }
            std::string tools_json = arr.dump();
            safe_copy(tools_json, tools_buf, tools_len, &truncated);
        } catch (...) {
            if (tools_buf && tools_len >= 3) { memcpy(tools_buf, "[]", 3); }
        }
        if (out_has_tools) *out_has_tools = 1;
    }
    return truncated ? 1 : 0;
}
