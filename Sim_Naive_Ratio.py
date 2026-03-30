# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
@author: Yikun Zhang
Last Editing: Mar 24, 2026

Description: Simulation on data with both concept and covariate shifts.
It contains XGBoost, kernel ridge regression, and neural network models
applied to the source-only and naively combined data.
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
        X_comb = np.concatenate([dat_source[i][:, 1:] for i in range(len(dat_source))] + [X0], axis=0)
        Y_comb = np.concatenate([dat_source[i][:, 0] for i in range(len(dat_source))] + [Y0], axis=0)
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
        grid_search.fit(X_comb, Y_comb)
        target_only_xgb = grid_search.best_estimator_
        xbg_comb = np.mean(abs(target_only_xgb.predict(X_test) - Y_test)**2)

        ## Kernel Ridge Regression
        alpha_lst = (0.1 / X_comb.shape[0] * (3.0 ** np.array(range(-2,6))))
        param_grid = {'alpha': alpha_lst}
        target_only_krr = KernelRidge(kernel='rbf')
        grid_search = GridSearchCV(target_only_krr, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_comb, Y_comb)
        target_only_krr = grid_search.best_estimator_
        krr_comb = np.mean(abs(target_only_krr.predict(X_test) - Y_test)**2)

        ## Neural Network
        param_grid = {
            'hidden_layer_sizes': [(10,), (50,), (100,)],
            'alpha': [0.0001, 0.001, 0.01],
        }
        mlp = MLPRegressor(max_iter=1000, random_state=0)
        grid_search = GridSearchCV(mlp, param_grid, cv=5)
        grid_search.fit(X_comb, Y_comb)
        target_only_mlp = grid_search.best_estimator_
        nn_comb = np.mean(abs(target_only_mlp.predict(X_test) - Y_test)**2)

        # ML models on a single-source data (the first source)
        X_source1 = dat_source[0][:, 1:]
        Y_source1 = dat_source[0][:, 0]

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
        grid_search.fit(X_source1, Y_source1)
        source_only_xgb = grid_search.best_estimator_
        xbg_source1 = np.mean(abs(source_only_xgb.predict(X_test) - Y_test)**2)

        ## Kernel Ridge Regression
        alpha_lst = (0.1 / X_source1.shape[0] * (3.0 ** np.array(range(-2,6))))
        param_grid = {'alpha': alpha_lst}
        target_only_krr = KernelRidge(kernel='rbf')
        grid_search = GridSearchCV(target_only_krr, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_source1, Y_source1)
        target_only_krr = grid_search.best_estimator_
        krr_source1 = np.mean(abs(target_only_krr.predict(X_test) - Y_test)**2)

        ## Neural Network
        param_grid = {
            'hidden_layer_sizes': [(10,), (50,), (100,)],
            'alpha': [0.0001, 0.001, 0.01],
        }
        mlp = MLPRegressor(max_iter=1000, random_state=0)
        grid_search = GridSearchCV(mlp, param_grid, cv=5)
        grid_search.fit(X_source1, Y_source1)
        source_only_mlp = grid_search.best_estimator_
        nn_source1 = np.mean(abs(source_only_mlp.predict(X_test) - Y_test)**2)

        # ML models on a single-source data (the second source)
        X_source2 = dat_source[1][:, 1:]
        Y_source2 = dat_source[1][:, 0]
        
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
        grid_search.fit(X_source2, Y_source2)
        source_only_xgb = grid_search.best_estimator_
        xbg_source2 = np.mean(abs(source_only_xgb.predict(X_test) - Y_test)**2)

        ## Kernel Ridge Regression
        alpha_lst = (0.1 / X_source2.shape[0] * (3.0 ** np.array(range(-2,6))))
        param_grid = {'alpha': alpha_lst}
        target_only_krr = KernelRidge(kernel='rbf')
        grid_search = GridSearchCV(target_only_krr, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_source2, Y_source2)
        target_only_krr = grid_search.best_estimator_
        krr_source2 = np.mean(abs(target_only_krr.predict(X_test) - Y_test)**2)

        ## Neural Network
        param_grid = {
            'hidden_layer_sizes': [(10,), (50,), (100,)],
            'alpha': [0.0001, 0.001, 0.01],
        }
        mlp = MLPRegressor(max_iter=1000, random_state=0)
        grid_search = GridSearchCV(mlp, param_grid, cv=5)
        grid_search.fit(X_source2, Y_source2)
        source_only_mlp = grid_search.best_estimator_
        nn_source2 = np.mean(abs(source_only_mlp.predict(X_test) - Y_test)**2)

        # Save results
        mse = np.array([xbg_comb, krr_comb, nn_comb, xbg_source1, krr_source1, nn_source1,
                        xbg_source2, krr_source2, nn_source2])
        res_names = ['XGBoost_Naive_Comb', 'KRR_Naive_Comb', 'NN_Naive_Comb',
                    'XGBoost_Source1', 'KRR_Source1', 'NN_Source1',
                    'XGBoost_Source2', 'KRR_Source2', 'NN_Source2']
        res_df = pd.DataFrame({'Method': res_names, 'MSE': mse})
        res_df['source_size'] = n_s
        res_df['target_size'] = n_0
        res_full = pd.concat([res_full, res_df], axis=0)

res_full.to_csv('./Results/Simulation_Concept_Covariate_'+str(job_id)+'_naive2.csv', index=False)


