# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
@author: Yikun Zhang
Last Editing: Jan 15, 2026

Description: Code for implementing our proposed TLCQM framework.
"""

import numpy as np
import torch
from engression import engression
from quantile_match import quantile_matching_estimate

#=======================================================================================#

def fit_TLCQM(dat_source, dat_target, X_dat_tensor=None, n_sampler=3000, random_state=None,
    # Engression model arguments
    eng_num_layer=2,
    eng_hidden_dim=100,
    eng_noise_dim=5,
    eng_lr=0.001,
    eng_num_epochs=1000,
    eng_pred_sample_size=500,
    # quantile matching arguments
    qm_beta_init=None,
    qm_stop_eps=1e-8,
    qm_max_iter=1000,
    qm_positive=False,
    qm_verbose=False):
    """
    Transfer Learning with Conditional Quantile Matching (TLCQM)

    Parameters
    ----------
        dat_source : list of np.ndarray
            Source datasets, each of shape (n_s, d+1).
        dat_target : np.ndarray
            Target labeled data of shape (n_0, d+1).
        X_dat_tensor : torch.Tensor or None
            Covariates where calibrated responses are produced.
        n_sampler : int
            Number of pseudo-samples drawn per source for quantile matching.
        random_state : int
            Random seed.

    Engression model arguments
    -------------------
        eng_num_layer, eng_hidden_dim, eng_noise_dim, eng_lr, eng_num_epochs
            Architecture and training hyperparameters of Engression.
        eng_pred_sample_size : int
            Number of Monte Carlo samples used for prediction.

    Quantile matching arguments
    ---------------------------
        qm_beta_init : np.ndarray or None
            Initial value for quantile matching coefficients.
        qm_stop_eps : float
            Convergence tolerance.
        qm_max_iter : int
            Maximum number of iterations.
        qm_verbose : bool
            Verbosity flag.
    Returns
    -------
        Y_matched : np.ndarray
            Calibrated responses for X_dat.
        beta_hat : np.ndarray
            Estimated quantile matching coefficients.
    """
    if random_state is not None:
        np.random.seed(random_state)
        torch.manual_seed(random_state)

    Y0 = dat_target[:, 0]
    X0 = dat_target[:, 1:]
    X0_tensor = torch.tensor(X0, dtype=torch.float32)

    # Fit Engression models on sources
    eng_models = []
    X_source_tensor = []
    for src in dat_source:
        Y_tensor = torch.tensor(src[:, 0].reshape(-1, 1), dtype=torch.float32)
        X_tensor = torch.tensor(src[:, 1:], dtype=torch.float32)

        engressor = engression(X_tensor, Y_tensor, num_layer=eng_num_layer, hidden_dim=eng_hidden_dim,
                         noise_dim=eng_noise_dim, lr=eng_lr, num_epochs=eng_num_epochs)
        X_source_tensor.append(X_tensor)
        eng_models.append(engressor)
    X_source_tensor = torch.cat(X_source_tensor, dim=0)
    if X_dat_tensor is None:
        X_dat_tensor = X_source_tensor

    # Sample pseudo-responses for target covariates
    Y0_sam = []
    for eng in eng_models:
        sam = (eng.sample(X0_tensor, sample_size=n_sampler).detach().numpy().reshape(-1, 1))
        Y0_sam.append(sam)
    Y0_sam = np.concatenate(Y0_sam, axis=1)
    Y0_sam_arr = np.concatenate([np.ones([Y0_sam.shape[0],1]), Y0_sam], axis=1)

    # Quantile matching
    beta_hat = quantile_matching_estimate(np.repeat(Y0, n_sampler), Y0_sam_arr, beta_init=qm_beta_init, 
                                          stop_eps=qm_stop_eps, max_iter=qm_max_iter, verbose=qm_verbose, 
                                          positive=qm_positive)

    # Predict source outputs for X_dat and calibrate
    Y_source_pred = []
    for eng in eng_models:
        pred = (
            eng.predict(X_dat_tensor, sample_size=eng_pred_sample_size).detach().numpy().reshape(-1, 1)
        )
        Y_source_pred.append(pred)

    Y_source_pred = np.concatenate(Y_source_pred, axis=1)
    Y_source_pred = np.concatenate([np.ones((Y_source_pred.shape[0], 1)), Y_source_pred], axis=1)

    # Apply quantile matching coefficients
    Y_matched = Y_source_pred @ beta_hat

    return Y_matched, beta_hat
