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
 * Hot-path note for `common_sampler`:
 *   `ion7_csampler_sample_accept` exists specifically so the per-token
 *   inner loop crosses the FFI boundary ONCE per token instead of
 *   twice. Keep using it from Lua unless you have a reason to inspect
 *   the sampled token before accepting it.
 */

#include "ion7_bridge.h"

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
 * ======================================================================= */

/**
 * Owning wrapper around `common_sampler*`. We hold the pointer in a
 * dedicated struct rather than reinterpret-casting the `common_sampler`
 * itself so we can later add per-instance state (e.g. counters,
 * cached parameters) without breaking the existing handle ABI.
 */
struct ion7_csampler {
    common_sampler* smpl;
};

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
int32_t ion7_csampler_sample_accept(ion7_csampler_t*      s,
                                    struct llama_context* ctx,
                                    int                   idx,
                                    int                   grammar_first)
{
    if (!s) return -1;
    llama_token tok = common_sampler_sample(s->smpl, ctx, idx, (bool)grammar_first);
    common_sampler_accept(s->smpl, tok, /*accept_grammar=*/true);
    return tok;
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

    w->params.type  = (common_speculative_type)type;
    w->params.n_max = (n_draft  > 0) ? n_draft : 16;
    w->params.n_min = 0;
    if (ngram_min > 0) w->params.ngram_size_n = (uint16_t)ngram_min;
    if (ngram_max > 0) w->params.ngram_size_m = (uint16_t)ngram_max;

    /* Draft-model branch : libcommon spins up its own draft context
     * internally via `llama_init_from_model(model_dft, cparams_dft)`.
     * The default-constructed `cparams_dft` is zero-filled — leaving
     * it untouched would request `n_ctx = 0`, which crashes the draft
     * context build. We seed it with llama.cpp's documented defaults
     * and mirror the target context's window size so the draft can
     * fit the same prompt. */
    if (ctx_dft) {
        w->params.model_dft   = const_cast<llama_model*>(llama_get_model(ctx_dft));
        w->params.cparams_dft = llama_context_default_params();
        if (ctx_tgt) {
            w->params.cparams_dft.n_ctx   = llama_n_ctx  (ctx_tgt);
            w->params.cparams_dft.n_batch = llama_n_batch(ctx_tgt);
        }
        /* libcommon's `has_draft` gate is `!params.mparams_dft.path.empty()`.
         * The path is only used to decide whether to register the
         * DRAFT implementation — `model_dft` is what actually drives
         * the draft pass. We seed a sentinel so the gate fires even
         * when the model has been loaded out-of-band. */
        w->params.mparams_dft.path = "<external>";
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
