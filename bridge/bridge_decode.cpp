/*
 * bridge_decode.cpp - JIT-friendly llama_decode / llama_encode shims.
 *
 * Copyright (C) 2026 Ion7 Project Contributors
 * SPDX-License-Identifier: MIT
 *
 * ──────────────────────────────────────────────────────────────────────────
 * This is the single intentional exception to the bridge's "libcommon-only"
 * rule (see ion7_bridge.h, top-of-file scope comment).
 *
 * `llama_decode` and `llama_encode` take `struct llama_batch` BY VALUE in
 * upstream llama.h. LuaJIT's FFI cannot JIT-compile C function calls that
 * pass or return aggregate types by value — the trace aborts with
 *   "NYI: unsupported C function type"
 * on every call, dropping the surrounding Lua loop back into the
 * interpreter. For a per-token generation loop that's catastrophic :
 * we measured the bailout cost as ~150 ns/token plus a snapshot reload
 * each time the trace stitches back. Across a 64-token run that's ~10 µs
 * of overhead and a roughly 2× stddev inflation.
 *
 * These shims pass the batch by pointer at the FFI boundary and
 * dereference inside C. The Lua hot loop now compiles into one
 * uninterrupted machine-code trace.
 *
 * No information is hidden ; the shims are a 1:1 contract over upstream.
 * ──────────────────────────────────────────────────────────────────────────
 */

#include "ion7_bridge.h"

extern "C" {

int32_t ion7_context_decode(struct llama_context * ctx, const struct llama_batch * batch) {
    return llama_decode(ctx, *batch);
}

int32_t ion7_context_encode(struct llama_context * ctx, const struct llama_batch * batch) {
    return llama_encode(ctx, *batch);
}

}  /* extern "C" */
