# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
@author: Yikun Zhang
Last Editing: Mar 24, 2026

Description: Simulation on data with both concept and covariate shifts.
It contains XGBoost, kernel ridge regression, and neural network models
applied to the mixup data (https://arxiv.org/pdf/1710.09412).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from utils import sim_data
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
import sys

job_id = int(sys.argv[1])
print(job_id)

#=======================================================================================#

def make_mixup_regression(X, y, alpha=0.4, n_aug=None, random_state=0):
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    if n_aug is None:
        n_aug = n

    idx1 = rng.integers(0, n, size=n_aug)
    idx2 = rng.integers(0, n, size=n_aug)
    lam = rng.beta(alpha, alpha, size=n_aug)

    X_mix = lam[:, None] * X[idx1] + (1.0 - lam)[:, None] * X[idx2]
    y_mix = lam * y[idx1] + (1.0 - lam) * y[idx2]
    return X_mix, y_mix


res_full = pd.DataFrame()
for n_0 in [50, 100, 150]:
    for fac_s in [1, 2, 5, 10, 15, 20, 30]:
        n_s = fac_s * n_0
        d = 5
        np.random.seed(job_id)
        dat_source, dat0, dat0_full, dat_test = sim_data(n_s=n_s, n_0=n_0, n_test=3000, sig=0.5, 
                                                        mu_s=np.ones(d), mu_t=np.zeros(d), Sigma=np.eye(d), 
                                                        beta1=1/np.arange(1, d+1))
        
        # ML models on a naive combination of source and target data
        X0 = dat0[:, 1:]
        Y0 = dat0[:, 0]
        for alpha in [0.2, 0.4, 0.5, 0.6, 0.8]:
            X_mix, Y_mix = make_mixup_regression(X0, Y0, alpha=alpha, n_aug=5 * len(Y0), random_state=job_id)
            X0_aug = np.vstack([X0, X_mix])
            Y0_aug = np.concatenate([Y0, Y_mix])
            X_test = dat_test[:, 1:]
            Y_test = dat_test[:, 0]

            ## XGBoost
            param_grid = {
                'learning_rate': [0.001, 0.01, 0.1],
                'n_estimators': [10, 50, 100], 
                'max_depth': [3, 5],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
            }
            xgb_model = XGBRegressor(objective='reg:squarederror', random_state=0)
            grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='neg_mean_squared_error')
            grid_search.fit(X0_aug, Y0_aug)
            target_only_xgb = grid_search.best_estimator_
            xbg_comb = np.mean(abs(target_only_xgb.predict(X_test) - Y_test)**2)

            ## Kernel Ridge Regression
            alpha_lst = (0.1 / X0_aug.shape[0] * (3.0 ** np.array(range(-2,6))))
            param_grid = {'alpha': alpha_lst}
            target_only_krr = KernelRidge(kernel='rbf')
            grid_search = GridSearchCV(target_only_krr, param_grid, cv=5, scoring='neg_mean_squared_error')
            grid_search.fit(X0_aug, Y0_aug)
            target_only_krr = grid_search.best_estimator_
            krr_comb = np.mean(abs(target_only_krr.predict(X_test) - Y_test)**2)

            ## Neural Network
            param_grid = {
                'hidden_layer_sizes': [(10,), (50,), (100,)],
                'alpha': [0.0001, 0.001, 0.01],
            }
            mlp = MLPRegressor(max_iter=1000, random_state=0)
            grid_search = GridSearchCV(mlp, param_grid, cv=5)
            grid_search.fit(X0_aug, Y0_aug)
            target_only_mlp = grid_search.best_estimator_
            nn_comb = np.mean(abs(target_only_mlp.predict(X_test) - Y_test)**2)

            # Save results
            mse = np.array([xbg_comb, krr_comb, nn_comb])
            res_names = ['XGBoost_Mixup_Comb', 'KRR_Mixup_Comb', 'NN_Mixup_Comb']
            res_df = pd.DataFrame({'Method': res_names, 'MSE': mse})
            res_df['source_size'] = n_s
            res_df['target_size'] = n_0
            res_df['alpha'] = alpha
            res_full = pd.concat([res_full, res_df], axis=0)

res_full.to_csv('./Results/Simulation_Concept_Covariate_'+str(job_id)+'_mixup2.csv', index=False)


