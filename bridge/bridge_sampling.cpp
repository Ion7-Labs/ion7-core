/*
 * bridge_sampling.cpp - common_sampler + common_speculative wrappers.
 *
 * Copyright (C) 2026 Ion7 Project Contributors
 * SPDX-License-Identifier: MIT
 *
 * Both APIs share the property that they live in libcommon's C++ layer
 * and would not survive a direct LuaJIT FFI binding (they expose
 * `std::vector` and `std::string` parameters and consume nlohmann/json
 * internally). The wrappers below pin them to a flat C ABI.
 *
 * For the per-token hot loop, prefer `ion7_csampler_sample_accept` (in
 * bridge_fast.cpp) which fuses sample + accept into a single FFI call.
 */

#include "ion7_bridge.h"
#include "bridge_internal.hpp"

#include "common.h"
#include "sampling.h"
#include "speculative.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <new>
#include <vector>

/* =========================================================================
 * common_sampler — DRY, XTC, Mirostat, grammar_lazy, ...
 *
 * `struct ion7_csampler` is defined in bridge_internal.hpp so the
 * sample_accept fast-path in bridge_fast.cpp can reach `s->smpl`.
 * ======================================================================= */

extern "C"
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
        sp.seed              = p->seed;
        sp.top_k             = p->top_k;
        sp.top_p             = p->top_p;
        sp.min_p             = p->min_p;
        sp.xtc_probability   = p->xtc_probability;
        sp.xtc_threshold     = p->xtc_threshold;
        sp.temp              = p->temp;
        sp.penalty_repeat    = p->repeat_penalty;
        sp.penalty_freq      = p->freq_penalty;
        sp.penalty_present   = p->pres_penalty;
        sp.penalty_last_n    = p->repeat_last_n;
        sp.dry_multiplier    = p->dry_mult;
        sp.dry_base          = p->dry_base;
        sp.dry_allowed_length = p->dry_allowed_len;
        sp.dry_penalty_last_n = p->dry_last_n;
        sp.mirostat          = p->mirostat;
        sp.mirostat_tau      = p->mirostat_tau;
        sp.mirostat_eta      = p->mirostat_eta;
        sp.top_n_sigma       = p->top_n_sigma;
        sp.adaptive_target   = p->adaptive_target;
        sp.adaptive_decay    = p->adaptive_decay;
        sp.grammar_lazy      = (bool)p->grammar_lazy;
    }

    if (grammar && grammar[0] != '\0') {
        sp.grammar = common_grammar(COMMON_GRAMMAR_TYPE_USER, grammar);
    }

    for (int i = 0; i < n_triggers && trigger_words; i++) {
        if (!trigger_words[i]) continue;
        common_grammar_trigger trig;
        trig.type  = COMMON_GRAMMAR_TRIGGER_TYPE_WORD;
        trig.value = trigger_words[i];
        sp.grammar_triggers.push_back(std::move(trig));
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
    if (!w->smpl) {
        delete w;
        return nullptr;
    }
    return w;
}

extern "C"
void ion7_csampler_free(ion7_csampler_t* s)
{
    if (!s) return;
    common_sampler_free(s->smpl);
    delete s;
}

extern "C"
int32_t ion7_csampler_sample(ion7_csampler_t*      s,
                             struct llama_context* ctx,
                             int                   idx,
                             int                   grammar_first)
{
    if (!s) return -1;
    return common_sampler_sample(s->smpl, ctx, idx, (bool)grammar_first);
}

extern "C"
void ion7_csampler_accept(ion7_csampler_t* s, int32_t token)
{
    if (s) common_sampler_accept(s->smpl, token, /*accept_grammar=*/true);
}

extern "C"
void ion7_csampler_reset(ion7_csampler_t* s)
{
    if (s) common_sampler_reset(s->smpl);
}

extern "C"
int32_t ion7_csampler_last(const ion7_csampler_t* s)
{
    if (!s) return -1;
    /* `common_sampler_last` throws when its ring buffer is empty;
     * we collapse that into a -1 sentinel for the C ABI. */
    try {
        return common_sampler_last(s->smpl);
    } catch (...) {
        return -1;
    }
}

extern "C"
uint32_t ion7_csampler_get_seed(const ion7_csampler_t* s)
{
    if (!s) return 0;
    return common_sampler_get_seed(s->smpl);
}

/* =========================================================================
 * Speculative decoding — common_speculative
 * ======================================================================= */

/**
 * Owning wrapper around `common_speculative*` plus its parameters and
 * a reusable token buffer. Keeping `toks_buf` resident avoids
 * reallocation every time `begin()` or `draft()` is called from Lua.
 */
struct ion7_speculative {
    common_speculative*       spec;
    common_params_speculative params;
    llama_tokens              toks_buf;
};

extern "C"
ion7_speculative_t* ion7_speculative_init(
    struct llama_context* ctx_tgt,
    struct llama_context* ctx_dft,
    int type, int n_draft, int ngram_min, int ngram_max)
{
    auto* w = new (std::nothrow) ion7_speculative;
    if (!w) return nullptr;

    w->params.type        = (common_speculative_type)type;
    w->params.draft.n_max = (n_draft > 0) ? n_draft : 16;
    w->params.draft.n_min = 0;

    /* ngram window applies to whichever speculator variant the caller
     * selected ; we write the same size to all three n-gram structs so
     * the active one picks it up regardless. */
    if (ngram_min > 0) {
        w->params.ngram_simple .size_n = (uint16_t)ngram_min;
        w->params.ngram_map_k  .size_n = (uint16_t)ngram_min;
        w->params.ngram_map_k4v.size_n = (uint16_t)ngram_min;
    }
    if (ngram_max > 0) {
        w->params.ngram_simple .size_m = (uint16_t)ngram_max;
        w->params.ngram_map_k  .size_m = (uint16_t)ngram_max;
        w->params.ngram_map_k4v.size_m = (uint16_t)ngram_max;
    }

    /* Seed `draft.cparams` with sane defaults mirroring the target
     * context — `common_speculative_init` calls `llama_init_from_model`
     * with these and a zero-filled struct would request n_ctx=0. */
    if (ctx_dft) {
        w->params.draft.model   = const_cast<llama_model*>(llama_get_model(ctx_dft));
        w->params.draft.cparams = llama_context_default_params();
        if (ctx_tgt) {
            w->params.draft.cparams.n_ctx   = llama_n_ctx  (ctx_tgt);
            w->params.draft.cparams.n_batch = llama_n_batch(ctx_tgt);
        }
        /* `has_draft()` checks `!draft.mparams.path.empty()` — we set
         * a sentinel so the gate fires when the draft model is supplied
         * out-of-band rather than loaded from a path. */
        w->params.draft.mparams.path = "<external>";
    }

    w->spec = common_speculative_init(w->params, ctx_tgt);
    if (!w->spec) {
        delete w;
        return nullptr;
    }
    return w;
}

extern "C"
void ion7_speculative_free(ion7_speculative_t* spec)
{
    if (!spec) return;
    common_speculative_free(spec->spec);
    delete spec;
}

extern "C"
void ion7_speculative_begin(ion7_speculative_t* spec,
                            const int32_t*      prompt,
                            int                 n_prompt)
{
    if (!spec || !prompt) return;
    spec->toks_buf.assign(prompt, prompt + n_prompt);
    common_speculative_begin(spec->spec, spec->toks_buf);
}

extern "C"
int ion7_speculative_draft(ion7_speculative_t* spec,
                           const int32_t* prompt, int n_prompt,
                           int32_t last_token,
                           int32_t* out_draft, int max_draft)
{
    if (!spec || !prompt || !out_draft) return 0;
    spec->toks_buf.assign(prompt, prompt + n_prompt);
    llama_tokens draft = common_speculative_draft(
        spec->spec, spec->params, spec->toks_buf, last_token);
    int n = (int)std::min((size_t)max_draft, draft.size());
    std::memcpy(out_draft, draft.data(), (size_t)n * sizeof(int32_t));
    return n;
}

extern "C"
void ion7_speculative_accept(ion7_speculative_t* spec, int n_accepted)
{
    if (spec) common_speculative_accept(spec->spec, (uint16_t)n_accepted);
}

extern "C"
void ion7_speculative_stats(const ion7_speculative_t* spec)
{
    if (spec) common_speculative_print_stats(spec->spec);
}
