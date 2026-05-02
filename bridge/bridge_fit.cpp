/*
 * bridge_fit.cpp - Bridge metadata + VRAM auto-fit.
 *
 * Copyright (C) 2026 Ion7 Project Contributors
 * SPDX-License-Identifier: MIT
 *
 * Two unrelated leaf functions live together here only because
 * neither warrants its own translation unit:
 *
 *   - `ion7_bridge_version`   : the bridge semver string.
 *   - `ion7_params_fit`       : VRAM auto-fit, wraps libcommon's
 *                               `common_fit_params` (was `llama_params_fit`
 *                               in older llama.cpp builds; the rename to
 *                               `common_*` happened when the function
 *                               graduated from the public `llama.h` to
 *                               the libcommon helper layer).
 *
 * The auto-fit wrapper takes care of allocating the scratch buffers
 * (`tensor_split`, `tensor_buft_overrides`, `margins`) that libcommon
 * needs but which would be awkward to size from Lua.
 */

#include "ion7_bridge.h"

#include "fit.h"          /* common_fit_params + COMMON_PARAMS_FIT_STATUS_* */
#include "ggml.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>

/* =========================================================================
 * Version
 * ======================================================================= */

extern "C"
const char* ion7_bridge_version(void)
{
    /* Bumped to 2.0.0 with the libcommon-only rewrite (was 1.1.0). */
    return "2.0.0";
}

/* =========================================================================
 * VRAM auto-fit
 * ======================================================================= */

extern "C"
int ion7_params_fit(const char* path,
                    int32_t*    n_gpu_layers,
                    uint32_t*   n_ctx,
                    uint32_t    n_ctx_min)
{
    if (!path || !n_gpu_layers || !n_ctx) return 2;  /* COMMON_..._ERROR */

    struct llama_model_params   mparams = llama_model_default_params();
    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = *n_ctx;

    /* libcommon expects three writable scratch buffers sized after the
     * runtime device topology. We compute the sizes once via the public
     * llama.h queries and free everything before returning. */
    const size_t max_dev = llama_max_devices();
    const size_t max_ovr = llama_max_tensor_buft_overrides();

    float*  tensor_split = (float*)std::calloc(max_dev, sizeof(float));
    auto*   buft_overrides = (struct llama_model_tensor_buft_override*)std::calloc(
        max_ovr, sizeof(struct llama_model_tensor_buft_override));
    size_t* margins = (size_t*)std::calloc(max_dev, sizeof(size_t));

    if (!tensor_split || !buft_overrides || !margins) {
        std::free(tensor_split);
        std::free(buft_overrides);
        std::free(margins);
        return 2;  /* hard allocation failure → ERROR */
    }

    enum common_params_fit_status status = common_fit_params(
        path, &mparams, &cparams,
        tensor_split, buft_overrides, margins,
        n_ctx_min, GGML_LOG_LEVEL_WARN);

    if (status == COMMON_PARAMS_FIT_STATUS_SUCCESS) {
        *n_gpu_layers = mparams.n_gpu_layers;
        *n_ctx        = cparams.n_ctx;
    }

    std::free(tensor_split);
    std::free(buft_overrides);
    std::free(margins);

    /* The COMMON_* enum is intentionally 1:1 with our public ABI:
     *   0 = SUCCESS, 1 = FAILURE (cannot fit), 2 = ERROR. */
    return (int)status;
}
