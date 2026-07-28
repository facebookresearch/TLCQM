# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
@author: Yikun Zhang
Last Editing: Jul 25, 2026

Description: Code for the revised conditional TLCQM framework.
"""

import numpy as np
import torch
from engression import engression
from quantile_match_conditional import conditional_crps_estimate

#=======================================================================================#


def _sample_models(eng_mod, X_tensor, sample_size):
    n = X_tensor.shape[0]
    Y_sample = []
    for eng in eng_mod:
        sample = eng.sample(X_tensor, sample_size=sample_size)
        Y_sample.append(sample.detach().cpu().numpy().reshape(n, sample_size))
    return np.stack(Y_sample, axis=2)


def fit_TLCQM_conditional(
    dat_source,
    dat_target,
    X_dat_tensor=None,
    n_sampler=3000,
    random_state=None,
    # Engression model arguments
    eng_num_layer=2,
    eng_hidden_dim=100,
    eng_noise_dim=5,
    eng_lr=0.001,
    eng_num_epochs=1000,
    eng_pred_sample_size=500,
    eng_verbose=False,
    # Conditional matching arguments
    qm_beta_init=None,
    qm_stop_eps=1e-8,
    qm_max_iter=1000,
    qm_positive=False,
    qm_beta_bound=10.0,
    qm_n_restarts=10,
    qm_proximal=1e-4,
    qm_verbose=False,
    # Pseudo-label construction
    pseudo_label_mode="mean",
):
    """
    Transfer learning with covariate-indexed conditional CRPS matching.

    pseudo_label_mode="mean" uses calibrated conditional means for
    squared-error regression. The optional "sample" mode generates fresh
    calibrated responses for distributional prediction.
    """
    if random_state is not None:
        np.random.seed(random_state)
        torch.manual_seed(random_state)

    dat_target = np.asarray(dat_target, dtype=float)
    if dat_target.ndim != 2 or dat_target.shape[1] < 2:
        raise ValueError("dat_target must have shape (n_0, d+1).")
    if len(dat_source) == 0:
        raise ValueError("At least one source dataset is required.")
    if n_sampler < 2:
        raise ValueError("n_sampler must be at least two.")
    Y0 = dat_target[:, 0]
    X0_tensor = torch.tensor(dat_target[:, 1:], dtype=torch.float32)

    eng_mod = []
    X_source_tensor = []
    for dat in dat_source:
        dat = np.asarray(dat, dtype=float)
        if dat.ndim != 2 or dat.shape[1] != dat_target.shape[1]:
            raise ValueError("Every source must have shape (n_k, d+1).")
        Y_tensor = torch.tensor(dat[:, 0].reshape(-1, 1), dtype=torch.float32)
        X_tensor = torch.tensor(dat[:, 1:], dtype=torch.float32)
        eng = engression(
            X_tensor,
            Y_tensor,
            num_layer=eng_num_layer,
            hidden_dim=eng_hidden_dim,
            noise_dim=eng_noise_dim,
            lr=eng_lr,
            num_epochs=eng_num_epochs,
            verbose=eng_verbose,
        )
        eng_mod.append(eng)
        X_source_tensor.append(X_tensor)

    X_source_tensor = torch.cat(X_source_tensor, dim=0)
    if X_dat_tensor is None:
        X_dat_tensor = X_source_tensor
    elif not torch.is_tensor(X_dat_tensor):
        X_dat_tensor = torch.tensor(X_dat_tensor, dtype=torch.float32)
    else:
        X_dat_tensor = X_dat_tensor.detach().cpu().to(dtype=torch.float32)
    if X_dat_tensor.ndim != 2 or X_dat_tensor.shape[1] != X0_tensor.shape[1]:
        raise ValueError("X_dat_tensor must have shape (n, d).")

    Y0_sample = _sample_models(eng_mod, X0_tensor, n_sampler)
    beta_hat = conditional_crps_estimate(
        Y0,
        Y0_sample,
        beta_init=qm_beta_init,
        stop_eps=qm_stop_eps,
        max_iter=qm_max_iter,
        positive=qm_positive,
        beta_bound=qm_beta_bound,
        n_restarts=qm_n_restarts,
        proximal=qm_proximal,
        random_state=random_state,
        verbose=qm_verbose,
    )

    if pseudo_label_mode == "sample":
        Y_source = _sample_models(eng_mod, X_dat_tensor, 1)[:, 0, :]
    elif pseudo_label_mode == "mean":
        Y_source = []
        for eng in eng_mod:
            pred = eng.predict(
                X_dat_tensor, sample_size=eng_pred_sample_size
            )
            Y_source.append(pred.detach().cpu().numpy().reshape(-1, 1))
        Y_source = np.concatenate(Y_source, axis=1)
    else:
        raise ValueError("pseudo_label_mode must be 'sample' or 'mean'.")

    Y_source = np.column_stack([np.ones(Y_source.shape[0]), Y_source])
    Y_matched = Y_source @ beta_hat
    return Y_matched, beta_hat


fit_TLCQM = fit_TLCQM_conditional
