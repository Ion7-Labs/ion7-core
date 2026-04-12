/* bridge_training.cpp - training / fine-tuning via llama_opt API */
/*
 * Copyright (C) 2026 Ion7 Project Contributors
 * SPDX-License-Identifier: MIT
 */

#include "ion7_bridge.h"

/* libcommon - C++ layer */
#include "common.h"

#include "ggml-opt.h"

#include <cstdint>
#include <string>
#include <vector>

/* =========================================================================
 * ── Training (llama_opt API) ──────────────────────────────────────────── */

/* Userdata for the optimizer params callback - single constant learning rate. */
struct ion7_lr_params { float lr; };

/* ggml_opt_get_optimizer_params implementation: constant lr for AdamW and SGD. */
static ggml_opt_optimizer_params ion7_lr_callback(void* ud)
{
    ion7_lr_params* p = (ion7_lr_params*)ud;
    ggml_opt_optimizer_params op = ggml_opt_get_default_optimizer_params(ud);
    op.adamw.alpha = p->lr;
    op.sgd.alpha   = p->lr;
    return op;
}

/*
 * ion7_opt_init - initialise training on a context.
 *
 * optimizer : 0 = AdamW (recommended), 1 = SGD
 * lr        : learning rate (e.g. 1e-4 for LoRA, 1e-5 for full fine-tune)
 *
 * Must be called before ion7_opt_epoch.
 * The context must have been created without quantized KV cache (F16/F32).
 */
ion7_opt_state_t* ion7_opt_init(struct llama_context* ctx, struct llama_model* model, int optimizer, float lr)
{
    ion7_lr_params* lrp = new ion7_lr_params{ lr };

    struct llama_opt_params p;
    p.n_ctx_train    = 0;  /* use context size */
    p.param_filter   = llama_opt_param_filter_all;
    p.param_filter_ud = nullptr;
    p.get_opt_pars   = ion7_lr_callback;
    p.get_opt_pars_ud = (void*)lrp;
    p.optimizer_type = (optimizer == 1) ? GGML_OPT_OPTIMIZER_TYPE_SGD : GGML_OPT_OPTIMIZER_TYPE_ADAMW;

    llama_opt_init(ctx, model, p);
    return (ion7_opt_state_t*)lrp;  /* opaque handle to release later */
}

void ion7_opt_free(ion7_opt_state_t* state)
{
    delete (ion7_lr_params*)state;
}

/*
 * ion7_opt_dataset_create - build a training dataset from a token array.
 *
 * tokens   : flat array of llama_token (the full corpus, pre-tokenised)
 * n_tokens : number of tokens
 * stride   : number of tokens between the start of consecutive samples
 *            (pass llama_n_ctx(ctx) for non-overlapping windows)
 *
 * Returns an opaque dataset handle. Free with ion7_opt_dataset_free.
 */
ggml_opt_dataset_t ion7_opt_dataset_create(struct llama_context* ctx, const llama_token* tokens, int64_t n_tokens, int64_t stride)
{
    std::vector<llama_token> tok_vec(tokens, tokens + n_tokens);
    return common_opt_dataset_init(ctx, tok_vec, stride);
}

void ion7_opt_dataset_free(ggml_opt_dataset_t dataset)
{
    if (dataset) ggml_opt_dataset_free(dataset);
}

/*
 * ion7_opt_epoch - run one training epoch.
 *
 * val_split : fraction of dataset used for validation (0.0 = no validation)
 * Returns   : training loss for this epoch, or -1.0 on error.
 */
float ion7_opt_epoch(struct llama_context* ctx, ggml_opt_dataset_t dataset, float val_split)
{
    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_eval  = ggml_opt_result_init();

    int64_t n_data      = ggml_opt_dataset_data(dataset)->ne[1];
    int64_t idata_split = (int64_t)((double)n_data * (1.0 - (double)val_split));

    llama_opt_epoch(ctx, dataset, result_train, result_eval, idata_split, nullptr, nullptr);

    double loss = 0.0, uncertainty = 0.0;
    ggml_opt_result_loss(result_train, &loss, &uncertainty);

    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_eval);

    return (float)loss;
}
