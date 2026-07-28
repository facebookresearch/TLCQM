# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
@author: Yikun Zhang
Last Editing: Jul 24, 2026

Description: Source-to-target ratio simulation for conditional TLCQM.
"""

import sys

import numpy as np
import pandas as pd
from covariate_shift_conditional import kernel_mean_matching
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from TLCQM_conditional import fit_TLCQM_conditional
from utils import sim_data
from xgboost import XGBRegressor

job_id = int(sys.argv[1])
print(job_id)

#=======================================================================================#


def fit_models(X_train, Y_train, X_test, Y_test, weights=None):
    xgb_grid = {
        "learning_rate": [0.001, 0.01, 0.1],
        "n_estimators": [10, 50, 100],
        "max_depth": [3, 5],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    alpha = 0.1 / X_train.shape[0] * (3.0 ** np.arange(-2, 6))
    krr_grid = {"alpha": alpha}
    nn_grid = {
        "hidden_layer_sizes": [(10,), (50,), (100,)],
        "alpha": [0.0001, 0.001, 0.01],
    }

    model_all = [
        ("XGBoost", XGBRegressor(objective="reg:squarederror", random_state=0),
         xgb_grid),
        ("KRR", KernelRidge(kernel="rbf"), krr_grid),
        ("NN", MLPRegressor(max_iter=1000, random_state=0), nn_grid),
    ]
    mse = {}
    for name, model, param_grid in model_all:
        grid = GridSearchCV(
            model, param_grid, cv=5, scoring="neg_mean_squared_error"
        )
        grid.fit(X_train, Y_train, sample_weight=weights)
        pred = grid.best_estimator_.predict(X_test)
        mse[name] = np.mean((pred - Y_test) ** 2)
    return mse


n_0_all = [50, 100, 150]
fac_s_all = [1, 2, 5, 10, 15, 20, 30]
n_test = 3000
N_sam = 3000

res_full = []
for n_0 in n_0_all:
    for fac_s in fac_s_all:
        n_s = fac_s * n_0
        d = 5
        np.random.seed(job_id)
        dat_source, dat0, dat0_full, dat_test = sim_data(
            n_s=n_s,
            n_0=n_0,
            n_test=n_test,
            sig=0.5,
            mu_s=np.ones(d),
            mu_t=np.zeros(d),
            Sigma=np.eye(d),
            beta1=1 / np.arange(1, d + 1),
        )

        X0, Y0 = dat0[:, 1:], dat0[:, 0]
        X0_full, Y0_full = dat0_full[:, 1:], dat0_full[:, 0]
        X_test, Y_test = dat_test[:, 1:], dat_test[:, 0]
        X_source = np.concatenate([dat[:, 1:] for dat in dat_source], axis=0)

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

        kmm_weights = np.concatenate([
            kernel_mean_matching(
                X0, dat[:, 1:], kern="rbf", B=10
            )[:, 0]
            for dat in dat_source
        ])
        X_comb = np.concatenate([X_source, X0], axis=0)
        Y_comb = np.concatenate([Y_matched, Y0], axis=0)
        weights = np.concatenate([kmm_weights, np.ones(n_0)])
        res_tlcqm = fit_models(X_comb, Y_comb, X_test, Y_test, weights)

        for sample, result in [
            ("Target_Only", res_target),
            ("Oracle", res_oracle),
            ("TLCQM", res_tlcqm),
        ]:
            for model, mse in result.items():
                res_full.append({
                    "Method": model + "_" + sample,
                    "MSE": mse,
                    "source_target_ratio": fac_s,
                    "source_size": n_s,
                    "target_size": n_0,
                    "beta_hat": np.array2string(beta_hat, separator=","),
                })

pd.DataFrame(res_full).to_csv(
    "./Results/Simulation_Concept_Covariate_" + str(job_id)
    + "_conditional_mean.csv",
    index=False,
)
