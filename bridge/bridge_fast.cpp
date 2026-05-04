/*
 * bridge_fast.cpp - Composite fast-path bridge functions.
 *
 * Copyright (C) 2026 Ion7 Project Contributors
 * SPDX-License-Identifier: MIT
 *
 * ──────────────────────────────────────────────────────────────────────────
 * This file holds the upper tier of ion7-core's two-level C ABI :
 *
 *   - bridge_decode.cpp           1:1 wrappers around llama.cpp APIs
 *                                 that pass aggregates by value (the
 *                                 NYI shims for LuaJIT FFI). One C
 *                                 function per llama.cpp function.
 *
 *   - bridge_fast.cpp (this file) Composite shortcuts that fuse a
 *                                 common pipeline into a single bridge
 *                                 call. Same performance envelope as
 *                                 the primitives — LuaJIT's JIT already
 *                                 collapses sequences of FFI calls into
 *                                 one trace — but lower line count in
 *                                 user code.
 *
 * Pattern : every fast-path function here MUST be expressible as a
 * straightforward composition of primitives in `llama.h`. If a Lua
 * caller wants to inspect intermediate state, they drop back to the
 * primitive API ; the fast-path stays out of the way.
 * ──────────────────────────────────────────────────────────────────────────
 */

#include "ion7_bridge.h"
#include "bridge_internal.hpp"

#include "common.h"
#include "sampling.h"

extern "C" {

/* ── ion7_context_step : sample + EOG check + decode in one call ───────── */

int32_t ion7_context_step(struct llama_context     * ctx,
                          struct llama_batch       * batch,
                          struct llama_sampler     * sampler,
                          const  struct llama_vocab * vocab,
                          int                        idx)
{
    llama_token tok = llama_sampler_sample(sampler, ctx, idx);
    if (llama_vocab_is_eog(vocab, tok)) return -1;

    batch->token[0] = tok;
    if (llama_decode(ctx, *batch) != 0) return -2;

    return tok;
}

/* ── ion7_csampler_sample_accept : sample + accept in one call ─────────── */

int32_t ion7_csampler_sample_accept(ion7_csampler_t      * s,
                                    struct llama_context * ctx,
                                    int                    idx,
                                    int                    grammar_first)
{
    if (!s) return -1;
    llama_token tok = common_sampler_sample(s->smpl, ctx, idx, (bool)grammar_first);
    common_sampler_accept(s->smpl, tok, /*accept_grammar=*/true);
    return tok;
}

}  /* extern "C" */
