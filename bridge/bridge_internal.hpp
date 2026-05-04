#pragma once
/*
 * bridge_internal.hpp - Private helpers shared between bridge_*.cpp files.
 *
 * Copyright (C) 2026 Ion7 Project Contributors
 * SPDX-License-Identifier: MIT
 *
 * NOT part of the public API. Do not include from any external consumer
 * (Lua, downstream C++ libraries, etc.). Lives here only to keep the
 * implementation files free of duplicated tiny utilities.
 */

#include "ion7_bridge.h"

#include "common.h"
#include "sampling.h"

#include <cstring>
#include <cstdint>
#include <string>

/**
 * Concrete representation of `ion7_csampler_t`. Shared between
 * `bridge_sampling.cpp` (init / free / sample / accept / reset / ...)
 * and `bridge_fast.cpp` (sample_accept hot-path) so the fast-path can
 * reach `s->smpl` without going through an accessor indirection.
 */
struct ion7_csampler {
    common_sampler* smpl;
};

/**
 * Copy a `std::string` into a caller-provided C buffer with explicit
 * truncation handling. Used by parse-style bridge functions that emit
 * strings of unknown size into fixed-capacity buffers.
 *
 * The destination is always NUL-terminated (provided `cap > 0`).
 *
 * @param  src        Source string.
 * @param  dst        Destination buffer (caller-owned).
 * @param  cap        Capacity of `dst` in bytes (including NUL slot).
 * @param  truncated  Out-flag set to 1 if `src` was longer than `cap-1`.
 *                    Existing value is preserved on no-truncation, so the
 *                    caller can OR-accumulate across multiple calls.
 */
inline void ion7_safe_copy(const std::string& src,
                           char*              dst,
                           int32_t            cap,
                           int*               truncated)
{
    if (!dst || cap <= 0) return;
    size_t n = src.size();
    if ((int32_t)n >= cap) {
        n = (size_t)(cap - 1);
        if (truncated) *truncated = 1;
    }
    std::memcpy(dst, src.c_str(), n);
    dst[n] = '\0';
}
