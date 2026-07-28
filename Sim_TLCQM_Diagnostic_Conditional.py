# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
@author: Yikun Zhang
Last Editing: Jul 24, 2026

Description: Reviewer-requested diagnostic simulations for conditional TLCQM.
"""

import sys

import numpy as np
import pandas as pd
from covariate_shift_conditional import kernel_mean_matching
from quantile_match_conditional import conditional_crps_score
from scipy.stats import wasserstein_distance
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from TLCQM_conditional import fit_TLCQM_conditional
from xgboost import XGBRegressor

job_id = int(sys.argv[1])
print(job_id)

#=======================================================================================#


def regression_mean(X, scenario, domain):
    if scenario == "reversed":
        if domain == "target":
            return X[:, 0]
        if domain == "source1":
            return -X[:, 0]
        return X[:, 1]

    z = X @ (1 / np.arange(1, X.shape[1] + 1))
    if domain == "target":
        return np.sin(2 * z)
    if domain == "source1":
        return np.sin(z)
    return np.cos(z)


def sim_diagnostic(n_s, n_0, n_test, scenario, sig=0.3, d=5):
    def make_data(n, domain):
        X = np.random.multivariate_normal(np.zeros(d), np.eye(d), size=n)
        Y = regression_mean(X, scenario, domain) + sig * np.random.randn(n)
        return np.column_stack([Y, X])

    dat0 = make_data(n_0, "target")
    dat_source = [
        make_data(n_s, "source1"),
        make_data(n_s, "source2"),
    ]
    dat0_full = make_data(2 * n_s + n_0, "target")
    dat_test = make_data(n_test, "target")
    return dat_source, dat0, dat0_full, dat_test


def oracle_sample(X, scenario, M, sig, rng):
    sample = []
    for domain in ["source1", "source2"]:
        mean = regression_mean(X, scenario, domain)
        sample.append(mean[:, None] + sig * rng.normal(size=(len(X), M)))
    return np.stack(sample, axis=2)


def fit_models(X_train, Y_train, X_test, Y_test, weights=None):
    model_all = [
        ("XGBoost",
         XGBRegressor(objective="reg:squarederror", random_state=0),
         {
             "learning_rate": [0.001, 0.01, 0.1],
             "n_estimators": [10, 50, 100],
             "max_depth": [3, 5],
             "subsample": [0.8, 1.0],
             "colsample_bytree": [0.8, 1.0],
         }),
        ("KRR", KernelRidge(kernel="rbf"),
         {"alpha": 0.1 / X_train.shape[0]
          * (3.0 ** np.arange(-2, 6))}),
        ("NN", MLPRegressor(max_iter=1000, random_state=0),
         {
             "hidden_layer_sizes": [(10,), (50,), (100,)],
             "alpha": [0.0001, 0.001, 0.01],
         }),
    ]

    mse = {}
    for name, model, param_grid in model_all:
        grid = GridSearchCV(
            model, param_grid, cv=5, scoring="neg_mean_squared_error"
        )
        grid.fit(X_train, Y_train, sample_weight=weights)
        mse[name] = np.mean(
            (grid.best_estimator_.predict(X_test) - Y_test) ** 2
        )
    return mse


n_0_all = [50, 100, 150]
fac_s_all = [1, 2, 5, 10, 15, 20, 30]
scenario_all = ["reversed", "outside_span"]
n_test = 3000
N_sam = 3000
M_eval = 500
sig = 0.3

records = []
for scenario in scenario_all:
    for n_0 in n_0_all:
        for fac_s in fac_s_all:
            n_s = fac_s * n_0
            np.random.seed(job_id)
            dat_source, dat0, dat0_full, dat_test = sim_diagnostic(
                n_s, n_0, n_test, scenario, sig=sig
            )

            X0, Y0 = dat0[:, 1:], dat0[:, 0]
            X0_full, Y0_full = dat0_full[:, 1:], dat0_full[:, 0]
            X_test, Y_test = dat_test[:, 1:], dat_test[:, 0]
            X_source = np.concatenate(
                [dat[:, 1:] for dat in dat_source], axis=0
            )

            res_target = fit_models(X0, Y0, X_test, Y_test)
            res_oracle = fit_models(X0_full, Y0_full, X_test, Y_test)
            Y_matched, beta_hat = fit_TLCQM_conditional(
                dat_source,
                dat0,
                X_dat_tensor=X_source,
                n_sampler=N_sam,
                random_state=job_id,
                eng_num_epochs=1000,
                eng_pred_sample_size=200,
                qm_beta_bound=10,
                qm_n_restarts=10,
                pseudo_label_mode="mean",
            )

            kmm = np.concatenate([
                kernel_mean_matching(
                    X0, dat[:, 1:], kern="rbf", B=10
                )[:, 0]
                for dat in dat_source
            ])
            X_comb = np.concatenate([X_source, X0], axis=0)
            Y_comb = np.concatenate([Y_matched, Y0], axis=0)
            weights = np.concatenate([kmm, np.ones(n_0)])
            res_tlcqm = fit_models(
                X_comb, Y_comb, X_test, Y_test, weights
            )

            rng = np.random.default_rng(job_id + 10)
            target_eval = regression_mean(
                X_test, scenario, "target"
            )[:, None] + sig * rng.normal(size=(n_test, M_eval))
            source_eval = oracle_sample(
                X_test, scenario, M_eval, sig, rng
            )
            tlcqm_eval = (
                beta_hat[0]
                + np.einsum("nmk,k->nm", source_eval, beta_hat[1:])
            )
            mean_tlcqm = (
                beta_hat[0]
                + beta_hat[1] * regression_mean(
                    X_test, scenario, "source1"
                )
                + beta_hat[2] * regression_mean(
                    X_test, scenario, "source2"
                )
            )

            diagnostic = {
                "Marginal_W1_Source1": wasserstein_distance(
                    target_eval.ravel(), source_eval[:, :, 0].ravel()
                ),
                "Conditional_W1_Source1": np.mean(np.abs(
                    np.sort(target_eval, axis=1)
                    - np.sort(source_eval[:, :, 0], axis=1)
                )),
                "Conditional_W1_TLCQM": np.mean(np.abs(
                    np.sort(target_eval, axis=1)
                    - np.sort(tlcqm_eval, axis=1)
                )),
                "Conditional_CRPS_TLCQM": conditional_crps_score(
                    Y_test, source_eval, beta_hat
                ),
                "Conditional_Mean_RMSE_TLCQM": np.sqrt(np.mean(
                    (mean_tlcqm
                     - regression_mean(X_test, scenario, "target")) ** 2
                )),
            }

            for model in res_target:
                diagnostic["MSE_" + model + "_Target_Only"] = res_target[model]
                diagnostic["MSE_" + model + "_Oracle"] = res_oracle[model]
                diagnostic["MSE_" + model + "_TLCQM"] = res_tlcqm[model]
                diagnostic["Negative_Transfer_" + model] = (
                    res_tlcqm[model] - res_target[model]
                )

            for metric, value in diagnostic.items():
                records.append({
                    "Scenario": scenario,
                    "Metric": metric,
                    "Value": value,
                    "source_target_ratio": fac_s,
                    "source_size": n_s,
                    "target_size": n_0,
                    "beta_hat": np.array2string(beta_hat, separator=","),
                })

pd.DataFrame(records).to_csv(
    "./Results/Simulation_Diagnostic_" + str(job_id)
    + "_conditional_mean.csv",
    index=False,
)
