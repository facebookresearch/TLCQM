# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
@author: Yikun Zhang
Last Editing: Jul 24, 2026

Description: Ablation study of TLCQM on data with both concept and 
covariate shifts.It contains XGBoost, kernel ridge regression, and 
neural network models applied to the engression-only, simple mean 
matching, and 1D optimal transport data.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from utils import sim_data
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
import torch
import sys
from engression import engression
from quantile_match_conditional import conditional_crps_estimate
from covariate_shift_conditional import kernel_mean_matching

job_id = int(sys.argv[1])
print(job_id)

#=======================================================================================#

def wasserstein_1d(x, y):
    """
    Empirical 1D Wasserstein-1 distance between two samples.
    If lengths differ, compare empirical quantiles on a common grid.
    """
    x = np.sort(np.asarray(x).ravel())
    y = np.sort(np.asarray(y).ravel())

    m = len(x)
    n = len(y)
    q = np.linspace(0.0, 1.0, max(m, n), endpoint=False) + 0.5 / max(m, n)

    xq = np.quantile(x, q, method='linear')
    yq = np.quantile(y, q, method='linear')
    return np.mean(np.abs(xq - yq))


def pinball_loss(y_true, y_pred, tau=0.5):
    """
    Standard pinball / quantile loss for point predictions.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    diff = y_true - y_pred
    return np.mean(np.maximum(tau * diff, (tau - 1) * diff))


def avg_pinball_loss_distribution(y_true, y_sample, taus=None):
    """
    Compare a sample-based predictive distribution y_sample to y_true
    using averaged quantile loss over multiple taus.

    y_true: shape (n,)
    y_sample: shape (n, M) or (n*M,) if already flattened by observation
    """
    y_true = np.asarray(y_true).ravel()

    if taus is None:
        taus = np.linspace(0.1, 0.9, 9)

    if y_sample.ndim == 1:
        raise ValueError("y_sample should be shape (n, M) for distributional pinball loss.")

    losses = []
    for tau in taus:
        qhat = np.quantile(y_sample, tau, axis=1)
        losses.append(pinball_loss(y_true, qhat, tau=tau))
    return np.mean(losses)


def crps_ensemble(y_true, y_sample):
    """
    CRPS for ensemble/sample forecasts:
    CRPS(F, y) = E|X - y| - 0.5 E|X - X'|
    where X, X' ~ F iid.

    y_true: shape (n,)
    y_sample: shape (n, M)
    """
    y_true = np.asarray(y_true).ravel()
    y_sample = np.asarray(y_sample)

    _, M = y_sample.shape
    if M < 2:
        raise ValueError("At least two draws are required.")
    term1 = np.mean(np.abs(y_sample - y_true[:, None]), axis=1)
    y_sort = np.sort(y_sample, axis=1)
    rank = 2 * np.arange(M) - M + 1
    term2 = np.sum(y_sort * rank, axis=1) / (M * (M - 1))
    return np.mean(term1 - term2)

def fit_and_eval_models(X_train, y_train, X_test, y_test, sample_weight=None):
    out = {}

    # XGBoost
    param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'n_estimators': [10, 50, 100],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
    }
    xgb_model = XGBRegressor(objective='reg:squarederror', random_state=0)
    gs = GridSearchCV(xgb_model, param_grid, cv=5, scoring='neg_mean_squared_error')
    gs.fit(X_train, y_train, sample_weight=sample_weight)
    out['XGBoost'] = np.mean((gs.best_estimator_.predict(X_test) - y_test) ** 2)

    # KRR
    alpha_lst = 0.1 / X_train.shape[0] * (3.0 ** np.arange(-2, 6))
    gs = GridSearchCV(KernelRidge(kernel='rbf'),
                      {'alpha': alpha_lst},
                      cv=5,
                      scoring='neg_mean_squared_error')
    gs.fit(X_train, y_train, sample_weight=sample_weight)
    out['KRR'] = np.mean((gs.best_estimator_.predict(X_test) - y_test) ** 2)

    # NN
    param_grid = {
        'hidden_layer_sizes': [(10,), (50,), (100,)],
        'alpha': [0.0001, 0.001, 0.01],
    }
    gs = GridSearchCV(MLPRegressor(max_iter=1000, random_state=0), param_grid, cv=5)
    gs.fit(X_train, y_train, sample_weight=sample_weight)
    out['NN'] = np.mean((gs.best_estimator_.predict(X_test) - y_test) ** 2)

    return out


def fit_engression_models(dat_source):
    eng_mod = []
    X_source_tensor = []
    for src in dat_source:
        Y_tensor = torch.tensor(src[:, 0].reshape(-1, 1), dtype=torch.float32)
        X_tensor = torch.tensor(src[:, 1:], dtype=torch.float32)
        engressor = engression(
            X_tensor, Y_tensor,
            num_layer=2, hidden_dim=100, noise_dim=5,
            lr=0.001, num_epochs=1000
        )
        eng_mod.append(engressor)
        X_source_tensor.append(X_tensor)
    return eng_mod, torch.cat(X_source_tensor, dim=0)


def predict_means(eng_mod, X_tensor, sample_size=200):
    preds = []
    for mod in eng_mod:
        preds.append(mod.predict(X_tensor, sample_size=sample_size).detach().numpy().reshape(-1, 1))
    return np.concatenate(preds, axis=1)


def sample_responses(eng_mod, X_tensor, n_sam=3000):
    samps = []
    for mod in eng_mod:
        samps.append(mod.sample(X_tensor, sample_size=n_sam).detach().numpy().reshape(-1, 1))
    return np.concatenate(samps, axis=1)


def empirical_ot_map_1d(source_vals, target_vals):
    """
    Returns a callable T implementing the empirical 1D OT map
    T(u) = F_target^{-1}(F_source(u)).
    """
    source_sorted = np.sort(np.asarray(source_vals).ravel())
    target_sorted = np.sort(np.asarray(target_vals).ravel())
    n_s = len(source_sorted)
    n_t = len(target_sorted)

    def T1(u):
        u = np.asarray(u).ravel()
        # empirical CDF rank in source
        ranks = np.searchsorted(source_sorted, u, side='right') / n_s
        # map ranks to target quantiles
        idx = np.clip(np.floor(ranks * (n_t - 1)).astype(int), 0, n_t - 1)
        return target_sorted[idx]

    return T1


res_full = pd.DataFrame()
for n_0 in [50, 100, 150]:
    for fac_s in [1, 2, 5, 10, 15, 20, 30]:
        n_s = fac_s * n_0
        d = 5
        np.random.seed(job_id)
        dat_source, dat0, dat0_full, dat_test = sim_data(n_s=n_s, n_0=n_0, n_test=3000, sig=0.5, 
                                                        mu_s=np.ones(d), mu_t=np.zeros(d), Sigma=np.eye(d), 
                                                        beta1=1/np.arange(1, d+1))
        # dat_source, dat0, dat0_full, dat_test = sim_data2(n_s=n_s, n_0=n_0, n_test=3000, 
        #                                                 mu_s=np.ones(d), mu_t=np.zeros(d), Sigma=np.eye(d), 
        #                                                 beta1=1/np.arange(1, d+1), a=1.0, b=3.0)
        
        # =========================
        # Transfer variants / ablations
        # =========================
        X_dat0 = dat0[:, 1:]
        Y0 = dat0[:, 0]
        X_test = dat_test[:, 1:]
        Y_test = dat_test[:, 0]

        eng_mod, X_source_tensor = fit_engression_models(dat_source)
        X_dat0_tensor = torch.tensor(X_dat0, dtype=torch.float32)

        X_source = X_source_tensor.detach().numpy()

        # You currently use X_test in KMM; if you want training-time weighting,
        # X_dat0 is usually the more natural target sample here.
        kmm_weights = np.concatenate([
            kernel_mean_matching(
                X_dat0, dat[:, 1:], kern="rbf", B=10
            )[:, 0]
            for dat in dat_source
        ])

        # Predictive means on target/source covariates
        M_target = predict_means(eng_mod, X_dat0_tensor, sample_size=200)   # shape (n0, K)
        M_source = predict_means(eng_mod, X_source_tensor, sample_size=200) # shape (ns_total, K)

        # ----- (i) Engression only: no calibration -----
        Y_eng_only = M_source.mean(axis=1)
        calib_eng_only = {
            'W1_to_target': wasserstein_1d(Y_eng_only, Y0),
        }

        X_comb = np.concatenate([X_source, X_dat0], axis=0)
        Y_comb = np.concatenate([Y_eng_only, Y0], axis=0)
        weights = np.concatenate([kmm_weights, np.ones(X_dat0.shape[0])], axis=0)

        res_eng_only = fit_and_eval_models(X_comb, Y_comb, X_test, Y_test, sample_weight=weights)

        # ----- (ii) Mean matching: replace quantile matching by OLS on target means -----
        Z_target = np.concatenate([np.ones((M_target.shape[0], 1)), M_target], axis=1)
        Z_source = np.concatenate([np.ones((M_source.shape[0], 1)), M_source], axis=1)

        beta_mm, *_ = np.linalg.lstsq(Z_target, Y0, rcond=None)
        Y_mean_match = Z_source @ beta_mm
        calib_mean_match = {
            'W1_to_target': wasserstein_1d(Y_mean_match, Y0),
        }

        X_comb = np.concatenate([X_source, X_dat0], axis=0)
        Y_comb = np.concatenate([Y_mean_match, Y0], axis=0)
        weights = np.concatenate([kmm_weights, np.ones(X_dat0.shape[0])], axis=0)

        res_mean_match = fit_and_eval_models(X_comb, Y_comb, X_test, Y_test, sample_weight=weights)

        # ----- (iii) TLCQM: conditional CRPS matching -----
        N_sam = 3000
        Y0_sam = sample_responses(eng_mod, X_dat0_tensor, n_sam=N_sam)
        Y0_sam = Y0_sam.reshape(len(Y0), N_sam, len(eng_mod))
        beta_sol = conditional_crps_estimate(
            Y0,
            Y0_sam,
            beta_init=None,
            stop_eps=1e-8,
            max_iter=1000,
            beta_bound=10,
            n_restarts=10,
            random_state=job_id,
            verbose=False,
        )

        Z_source_qm = np.column_stack(
            [np.ones(M_source.shape[0]), M_source]
        )
        Y_tlcqm = Z_source_qm @ beta_sol

        X_comb = np.concatenate([X_source, X_dat0], axis=0)
        Y_comb = np.concatenate([Y_tlcqm, Y0], axis=0)
        weights = np.concatenate([kmm_weights, np.ones(X_dat0.shape[0])], axis=0)

        res_tlcqm = fit_and_eval_models(X_comb, Y_comb, X_test, Y_test, sample_weight=weights)

        # target-side calibrated samples for TLCQM
        Y_tlcqm_target_sam_mat = (
            beta_sol[0] + np.einsum("nmk,k->nm", Y0_sam, beta_sol[1:])
        )

        calib_tlcqm = {
            'W1_to_target': wasserstein_1d(
                Y_tlcqm_target_sam_mat.ravel(), np.repeat(Y0, N_sam)
            ),
            'AvgPinball': avg_pinball_loss_distribution(Y0, Y_tlcqm_target_sam_mat),
            'CRPS': crps_ensemble(Y0, Y_tlcqm_target_sam_mat),
        }

        # ----- (iv) OT calibration baseline (1D empirical OT) -----
        # Aggregate engression means first, then calibrate with 1D OT map
        target_score = M_target.mean(axis=1)
        source_score = M_source.mean(axis=1)

        T_ot = empirical_ot_map_1d(target_score, Y0)
        Y_ot = T_ot(source_score)
        calib_ot = {
        'W1_to_target': wasserstein_1d(Y_ot, Y0),
    }

        X_comb = np.concatenate([X_source, X_dat0], axis=0)
        Y_comb = np.concatenate([Y_ot, Y0], axis=0)
        weights = np.concatenate([kmm_weights, np.ones(X_dat0.shape[0])], axis=0)

        res_ot = fit_and_eval_models(X_comb, Y_comb, X_test, Y_test, sample_weight=weights)

        # Save results
        records = []

        method_metric_dict = {
            'Engression_Only': (res_eng_only, calib_eng_only),
            'Mean_Matching': (res_mean_match, calib_mean_match),
            'OT_Calibration': (res_ot, calib_ot),
            'TLCQM_Conditional': (res_tlcqm, calib_tlcqm),
        }

        for method_name, (pred_dict, calib_dict) in method_metric_dict.items():
            # downstream learner metrics
            for metric_name, value in pred_dict.items():
                records.append({
                    'Method': method_name,
                    'Metric': metric_name,
                    'Value': value,
                    'source_target_ratio': fac_s,
                    'source_size': n_s,
                    'target_size': n_0,
                })

            # calibration metrics
            for metric_name, value in calib_dict.items():
                records.append({
                    'Method': method_name,
                    'Metric': metric_name,
                    'Value': value,
                    'source_target_ratio': fac_s,
                    'source_size': n_s,
                    'target_size': n_0,
                })

        res_df = pd.DataFrame(records)
        res_full = pd.concat([res_full, res_df], axis=0)

res_full.to_csv(
    "./Results/Simulation_Concept_Covariate_" + str(job_id)
    + "_conditional_mean_abla.csv",
    index=False,
)
