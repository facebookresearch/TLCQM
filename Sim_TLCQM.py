# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
@author: Yikun Zhang
Last Editing: Jan 15, 2025

Description: Simulation on data with both concept and covariate shifts.
It contains XGBoost, kernel ridge regression, and neural network models
applied to the target-only, oracle, and TLCQM data.
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
from quantile_match import quantile_matching_estimate
from covariate_shift import kernel_mean_matching

job_id = int(sys.argv[1])
print(job_id)

#=======================================================================================#


res_full = pd.DataFrame()
for n_0 in [50, 100, 150, 200, 250]:
    for n_s in [100, 200, 500, 1000, 2000, 5000]:
        d = 5
        np.random.seed(job_id)
        dat_source, dat0, dat0_full, dat_test = sim_data(n_s=n_s, n_0=n_0, n_test=3000, sig=0.5, mu_s=np.ones(d), 
                                                         mu_t=np.zeros(d), Sigma=np.eye(d), beta1=1/np.arange(1, d+1))
        
        # Target-only ML models
        X0 = dat0[:, 1:]
        Y0 = dat0[:, 0]
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
        grid_search.fit(X0, Y0)
        target_only_xgb = grid_search.best_estimator_
        xbg_to = np.mean(abs(target_only_xgb.predict(X_test) - Y_test)**2)

        ## Kernel Ridge Regression
        alpha_lst = (0.1 / X0.shape[0] * (3.0 ** np.array(range(-2,6))))
        param_grid = {'alpha': alpha_lst}
        target_only_krr = KernelRidge(kernel='rbf')
        grid_search = GridSearchCV(target_only_krr, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X0, Y0)
        target_only_krr = grid_search.best_estimator_
        krr_to = np.mean(abs(target_only_krr.predict(X_test) - Y_test)**2)

        ## Neural Network
        param_grid = {
            'hidden_layer_sizes': [(10,), (50,), (100,)],
            'alpha': [0.0001, 0.001, 0.01],
        }
        mlp = MLPRegressor(max_iter=1000, random_state=0)
        grid_search = GridSearchCV(mlp, param_grid, cv=5)
        grid_search.fit(X0, Y0)
        target_only_mlp = grid_search.best_estimator_
        nn_to = np.mean(abs(target_only_mlp.predict(X_test) - Y_test)**2)


        # Oracle ML models
        X0_full = dat0_full[:, 1:]
        Y0_full = dat0_full[:, 0]

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
        grid_search.fit(X0_full, Y0_full)
        target_only_xgb = grid_search.best_estimator_
        xbg_ora = np.mean(abs(target_only_xgb.predict(X_test) - Y_test)**2)

        ## Kernel Ridge Regression
        alpha_lst = (0.1 / X0_full.shape[0] * (3.0 ** np.array(range(-2,6))))
        param_grid = {'alpha': alpha_lst}
        target_only_krr = KernelRidge(kernel='rbf')
        grid_search = GridSearchCV(target_only_krr, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X0_full, Y0_full)
        target_only_krr = grid_search.best_estimator_
        krr_ora = np.mean(abs(target_only_krr.predict(X_test) - Y_test)**2)

        ## Neural Network
        param_grid = {
            'hidden_layer_sizes': [(10,), (50,), (100,)],
            'alpha': [0.0001, 0.001, 0.01],
        }
        mlp = MLPRegressor(max_iter=1000, random_state=0)
        grid_search = GridSearchCV(mlp, param_grid, cv=5)
        grid_search.fit(X0_full, Y0_full)
        target_only_mlp = grid_search.best_estimator_
        nn_ora = np.mean(abs(target_only_mlp.predict(X_test) - Y_test)**2)


        # ML models with transfer learning with conditional quantile matching
        X_dat0 = dat0[:, 1:]
        Y0 = dat0[:, 0]

        # Fit the engression generative model on each source data
        eng_mod = []
        X_source_tensor = []
        for i in range(len(dat_source)):
            Y_tensor = torch.tensor(dat_source[i][:, 0].reshape(-1,1), dtype=torch.float32)
            X_tensor = torch.tensor(dat_source[i][:, 1:], dtype=torch.float32)
            engressor = engression(X_tensor, Y_tensor, num_layer=2, hidden_dim=100, noise_dim=5, lr=0.001, num_epochs=1000)
            X_source_tensor.append(X_tensor)
            eng_mod.append(engressor)
        X_dat0_tensor = torch.tensor(X_dat0, dtype=torch.float32)
        X_source_tensor = torch.cat(X_source_tensor, dim=0)

        # Sample response variables from each source data based on the covariates in the target domain
        N_sam = 3000
        Y0_sam = []
        for i in range(len(eng_mod)):
            Y0_sam.append(eng_mod[i].sample(X_dat0_tensor, sample_size=N_sam).detach().numpy().reshape(-1,1))
        Y0_sam = np.concatenate(Y0_sam, axis=1)
        Y0_sam_arr = np.concatenate([np.ones([Y0_sam.shape[0],1]), Y0_sam], axis=1)

        # Perform quantile matching to learn the adjustment coefficients
        beta_sol = quantile_matching_estimate(np.repeat(Y0, N_sam), Y0_sam_arr, beta_init=None, stop_eps=1e-8, max_iter=1000, verbose=False)

        Y_source_pred = []
        for i in range(len(eng_mod)):
            Y_source_pred.append(eng_mod[i].predict(X_source_tensor, sample_size=200).detach().numpy().reshape(-1,1))
        Y_source_pred = np.concatenate(Y_source_pred, axis=1)
        Y_source_pred = np.concatenate([np.ones([Y_source_pred.shape[0],1]), Y_source_pred], axis=1)
        Y_matched = np.dot(Y_source_pred, beta_sol)

        # Kernel mean matching for covariate shift correction
        X_source = X_source_tensor.detach().numpy()
        kmm_weights = kernel_mean_matching(X_test, X_source, kern='rbf', B=10)[:,0]

        X_comb = np.concatenate([X_source, X_dat0], axis=0)
        Y_comb = np.concatenate([Y_matched, Y0], axis=0)
        weights = np.concatenate([kmm_weights, np.ones(X_dat0.shape[0])], axis=0)

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
        grid_search.fit(X_comb, Y_comb, sample_weight=weights)
        target_only_xgb = grid_search.best_estimator_
        xbg_tlcqm = np.mean(abs(target_only_xgb.predict(X_test) - Y_test)**2)

        ## Kernel Ridge Regression
        alpha_lst = (0.1 / X_comb.shape[0] * (3.0 ** np.array(range(-2,6))))
        param_grid = {'alpha': alpha_lst}
        target_only_krr = KernelRidge(kernel='rbf')
        grid_search = GridSearchCV(target_only_krr, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_comb, Y_comb, sample_weight=weights)
        target_only_krr = grid_search.best_estimator_
        krr_tlcqm = np.mean(abs(target_only_krr.predict(X_test) - Y_test)**2)

        ## Neural Network
        param_grid = {
            'hidden_layer_sizes': [(10,), (50,), (100,)],
            'alpha': [0.0001, 0.001, 0.01],
        }
        mlp = MLPRegressor(max_iter=1000, random_state=0)
        grid_search = GridSearchCV(mlp, param_grid, cv=5)
        grid_search.fit(X_comb, Y_comb)
        target_only_mlp = grid_search.best_estimator_
        nn_tlcqm = np.mean(abs(target_only_mlp.predict(X_test) - Y_test)**2)

        # Save results
        mse = np.array([xbg_to, krr_to, nn_to, xbg_ora, krr_ora, nn_ora, xbg_tlcqm, krr_tlcqm, nn_tlcqm])
        res_names = ['XGBoost_Target_Only', 'KRR_Target_Only', 'NN_Target_Only',
                    'XGBoost_Oracle', 'KRR_Oracle', 'NN_Oracle',
                    'XGBoost_TLCQM', 'KRR_TLCQM', 'NN_TLCQM']
        res_df = pd.DataFrame({'Method': res_names, 'MSE': mse})
        res_df['source_size'] = n_s
        res_df['target_size'] = n_0
        res_full = pd.concat([res_full, res_df], axis=0)

res_full.to_csv('./Results/Simulation_Concept_Covariate_'+str(job_id)+'_new.csv', index=False)


