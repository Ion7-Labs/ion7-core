/*
 * bridge_training.cpp - Training loop wrapper around llama_opt + libcommon.
 *
 * Copyright (C) 2026 Ion7 Project Contributors
 * SPDX-License-Identifier: MIT
 *
 * Stays in the bridge for two reasons:
 *   - `common_opt_dataset_init` is libcommon-only (it materialises a
 *     `std::vector<llama_token>` and friends).
 *   - The `llama_opt_*` API uses C function-pointer callbacks for the
 *     learning-rate schedule. Routing those through LuaJIT FFI is
 *     possible but fragile (callback lifetimes, JIT trace aborts) and
 *     buys nothing here — the LR is a single scalar.
 *
 * The wrapper accepts a single constant LR for now; a future revision
 * can swap the callback for a richer schedule (cosine, warmup, ...)
 * without changing the public API.
 */

#include "ion7_bridge.h"

#include "common.h"
#include "ggml-opt.h"

#include <cstdint>
#include <new>
#include <vector>

/* =========================================================================
 * Optimiser-parameter callback (constant learning rate)
 * ======================================================================= */

/** Userdata captured by the LR callback — currently a single scalar. */
struct ion7_lr_params {
    float lr;
};

/**
 * `ggml_opt_get_optimizer_params` implementation that overrides only
 * the learning rate, leaving every other knob at the libcommon default.
 * The same `lr` is applied to both AdamW and SGD because we do not
 * know at registration time which optimiser ggml_opt will end up
 * using.
 */
static ggml_opt_optimizer_params ion7_lr_callback(void* ud)
{
    auto* p = static_cast<ion7_lr_params*>(ud);
    ggml_opt_optimizer_params op = ggml_opt_get_default_optimizer_params(ud);
    op.adamw.alpha = p->lr;
    op.sgd.alpha   = p->lr;
    return op;
}

/* =========================================================================
 * Training lifecycle
 * ======================================================================= */

extern "C"
ion7_opt_state_t* ion7_opt_init(struct llama_context* ctx,
                                struct llama_model*   model,
                                int                   optimizer,
                                float                 lr)
{
    auto* lrp = new (std::nothrow) ion7_lr_params{ lr };
    if (!lrp) return nullptr;

    struct llama_opt_params p;
    p.n_ctx_train     = 0;                          /* use the context size */
    p.param_filter    = llama_opt_param_filter_all;
    p.param_filter_ud = nullptr;
    p.get_opt_pars    = ion7_lr_callback;
    p.get_opt_pars_ud = static_cast<void*>(lrp);
    p.optimizer_type  = (optimizer == 1)
        ? GGML_OPT_OPTIMIZER_TYPE_SGD
        : GGML_OPT_OPTIMIZER_TYPE_ADAMW;

    llama_opt_init(ctx, model, p);
    /* The opaque handle is just the lr_params block — `llama_opt_init`
     * keeps the pointer alive internally for the lifetime of the
     * context's training session. We hand it back so the caller can
     * free it (and any future per-session state) via `ion7_opt_free`. */
    return reinterpret_cast<ion7_opt_state_t*>(lrp);
}

extern "C"
void ion7_opt_free(ion7_opt_state_t* state)
{
    delete reinterpret_cast<ion7_lr_params*>(state);
}

/* =========================================================================
 * Dataset construction
 * ======================================================================= */

extern "C"
ggml_opt_dataset_t ion7_opt_dataset_create(struct llama_context* ctx,
                                           const llama_token*    tokens,
                                           int64_t               n_tokens,
                                           int64_t               stride)
{
    /* libcommon expects a std::vector. We materialise once, here. */
    std::vector<llama_token> tok_vec(tokens, tokens + n_tokens);
    return common_opt_dataset_init(ctx, tok_vec, stride);
}

extern "C"
void ion7_opt_dataset_free(ggml_opt_dataset_t dataset)
{
    if (dataset) ggml_opt_dataset_free(dataset);
}

/* =========================================================================
 * Single-epoch driver
 * ======================================================================= */

extern "C"
float ion7_opt_epoch(struct llama_context* ctx,
                     ggml_opt_dataset_t    dataset,
                     float                 val_split)
{
    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_eval  = ggml_opt_result_init();

    /* Split position: first `idata_split` rows are training, the rest
     * are validation. `ne[1]` is the number of samples (rows). */
    int64_t n_data      = ggml_opt_dataset_data(dataset)->ne[1];
    int64_t idata_split = (int64_t)((double)n_data * (1.0 - (double)val_split));

    llama_opt_epoch(ctx, dataset, result_train, result_eval, idata_split,
                    /*callback_train=*/nullptr, /*callback_eval=*/nullptr);

    double loss = 0.0, uncertainty = 0.0;
    ggml_opt_result_loss(result_train, &loss, &uncertainty);

    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_eval);

    return (float)loss;
}
