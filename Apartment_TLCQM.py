#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang
Last Editing: Jan 2, 2025

Description: Application to the UCI Apartment for rent data.
It contains XGBoost, kernel ridge regression, and neural network models
applied to the target-only, oracle, and TLCQM data.
"""

import sys

import numpy as np
import pandas as pd
import torch
from covariate_shift import kernel_mean_matching
from engression import engression
from quantile_match import quantile_matching_estimate
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

job_id = int(sys.argv[1])
print(job_id)

# =======================================================================================#

# Read the data and preprocessing
data_raw = pd.read_csv(
    "data/apartments_for_rent_classified_100K.csv",
    encoding="latin1",
    engine="python",
    on_bad_lines="skip",
    sep=";",
)

# Filter the Outliers
data_ap = data_raw[data_raw.price.notna()]  # filter only those with a known price
data_ap = data_ap[data_ap.price_type == "Monthly"]
data_ap = data_ap[
    data_ap.price <= 50000
]  # removing an outlier studio apartment for 50K
data_ap = data_ap[data_ap.state.notna()]

# Filter to the Large States with at least 1000 listings
s = data_ap.state.astype(str).to_numpy(dtype=object)
SEGMENTS, counts = np.unique(s, return_counts=True)
SEGMENTS = SEGMENTS[counts >= 1000]
idx_large_states = np.where(data_ap.state.isin(SEGMENTS))[0]
data_ap = data_ap.iloc[idx_large_states, :]

# Column Subset
col_subset = [
    "state",
    "amenities",
    "bathrooms",
    "bedrooms",
    "has_photo",
    "pets_allowed",
    "square_feet",
    "price",
]
data_ap = data_ap.loc[:, col_subset]

# 4 Most common amenities converted to binary features
data_ap["Parking"] = data_ap["amenities"].str.contains("Parking", na=False).astype(int)
data_ap["Storage"] = data_ap["amenities"].str.contains("Storage", na=False).astype(int)
data_ap["Gym"] = data_ap["amenities"].str.contains("Gym", na=False).astype(int)
data_ap["Pool"] = data_ap["amenities"].str.contains("Pool", na=False).astype(int)

# Pets Allowed
data_ap["Cats"] = data_ap["pets_allowed"].str.contains("Cats", na=False).astype(int)
data_ap["Dogs"] = data_ap["pets_allowed"].str.contains("Dogs", na=False).astype(int)

# Fill in missing
data_ap.loc[data_ap["bathrooms"].isna(), "bathrooms"] = 0
data_ap.loc[data_ap["bedrooms"].isna(), "bedrooms"] = 0

source_domain = ["CA", "TX", "VA", "NC", "CO"]
target_domain = "FL"

# Prepare data for the source domains
dat_source = []
for s in source_domain:
    data_sub = data_ap.loc[
        data_ap["state"] == s,
        [
            "bathrooms",
            "bedrooms",
            "has_photo",
            "square_feet",
            "Parking",
            "Storage",
            "Gym",
            "Pool",
            "Cats",
            "Dogs",
        ],
    ]
    data_ap_transform = pd.get_dummies(data_sub, columns=["has_photo"], dtype=int)
    X_sub = data_ap_transform.values
    Y_sub = np.log(data_ap.loc[data_ap["state"] == s, "price"])
    dat1 = np.column_stack([Y_sub, X_sub])
    dat_source.append(dat1)

dat_pool = []
for s in source_domain:
    data_sub = data_ap.loc[
        data_ap["state"] == s,
        [
            "price",
            "state",
            "bathrooms",
            "bedrooms",
            "has_photo",
            "square_feet",
            "Parking",
            "Storage",
            "Gym",
            "Pool",
            "Cats",
            "Dogs",
        ],
    ]
    data_sub["price"] = np.log(data_sub["price"])
    dat_pool.append(data_sub)


res_full = pd.DataFrame()
for n_0 in [100, 200, 500, 1000, 2000]:
    data_sub = data_ap.loc[
        data_ap["state"] == target_domain,
        [
            "price",
            "state",
            "bathrooms",
            "bedrooms",
            "has_photo",
            "square_feet",
            "Parking",
            "Storage",
            "Gym",
            "Pool",
            "Cats",
            "Dogs",
        ],
    ]
    data_sub["price"] = np.log(data_sub["price"])
    dat0 = data_sub.sample(n=n_0, random_state=42)
    dat_pool.append(dat0)
    # Use the remaining rows for testing
    dat_test0 = data_sub.drop(dat0.index)

    dat0 = pd.get_dummies(
        dat0[
            [
                "price",
                "bathrooms",
                "bedrooms",
                "has_photo",
                "square_feet",
                "Parking",
                "Storage",
                "Gym",
                "Pool",
                "Cats",
                "Dogs",
            ]
        ],
        columns=["has_photo"],
        dtype=int,
    )
    dat0 = dat0.values
    dat_test0 = pd.get_dummies(
        dat_test0[
            [
                "price",
                "bathrooms",
                "bedrooms",
                "has_photo",
                "square_feet",
                "Parking",
                "Storage",
                "Gym",
                "Pool",
                "Cats",
                "Dogs",
            ]
        ],
        columns=["has_photo"],
        dtype=int,
    )
    dat_test0 = dat_test0

    dat_pool = pd.concat(dat_pool, axis=0)
    dat_pool = pd.get_dummies(dat_pool, columns=["has_photo"], dtype=int)
    n_pool = dat_pool.shape[0]
    dat_pool = pd.concat([dat_pool, dat_test0], axis=0).reset_index(drop=True)
    dat_pool = pd.get_dummies(dat_pool, columns=["state"], dtype=int)
    dat_pool_test0 = dat_pool.iloc[n_pool:, :].values
    dat_pool = dat_pool.iloc[:n_pool, :].values
    dat_test = dat_test0.values

    # Target-only ML models
    X0 = dat0[:, 1:]
    Y0 = dat0[:, 0]
    X_test = dat_test[:, 1:]
    Y_test = dat_test[:, 0]

    ## XGBoost
    param_grid = {
        "learning_rate": [0.001, 0.01, 0.1],
        "n_estimators": [10, 50, 100],
        "max_depth": [3, 5],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    xgb_model = XGBRegressor(objective="reg:squarederror", random_state=0)
    grid_search = GridSearchCV(
        xgb_model, param_grid, cv=5, scoring="neg_mean_squared_error"
    )
    grid_search.fit(X0, Y0)
    target_only_xgb = grid_search.best_estimator_
    xbg_to = np.mean(abs(target_only_xgb.predict(X_test) - Y_test) ** 2)

    ## Kernel Ridge Regression
    alpha_lst = 0.1 / X0.shape[0] * (3.0 ** np.array(range(-2, 6)))
    param_grid = {"alpha": alpha_lst}
    target_only_krr = KernelRidge(kernel="rbf")
    grid_search = GridSearchCV(
        target_only_krr, param_grid, cv=5, scoring="neg_mean_squared_error"
    )
    grid_search.fit(X0, Y0)
    target_only_krr = grid_search.best_estimator_
    krr_to = np.mean(abs(target_only_krr.predict(X_test) - Y_test) ** 2)

    ## Neural Network
    param_grid = {
        "hidden_layer_sizes": [(10,), (50,), (100,)],
        "alpha": [0.0001, 0.001, 0.01],
    }
    mlp = MLPRegressor(max_iter=1000, random_state=0)
    grid_search = GridSearchCV(mlp, param_grid, cv=5)
    grid_search.fit(X0, Y0)
    target_only_mlp = grid_search.best_estimator_
    nn_to = np.mean(abs(target_only_mlp.predict(X_test) - Y_test) ** 2)

    # ML models on Pooled Data
    X0_full = dat_pool[:, 1:]
    Y0_full = dat_pool[:, 0]
    X_test = dat_pool_test0[:, 1:]
    Y_test = dat_pool_test0[:, 0]

    ## XGBoost
    param_grid = {
        "learning_rate": [0.001, 0.01, 0.1],
        "n_estimators": [10, 50, 100],
        "max_depth": [3, 5],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    xgb_model = XGBRegressor(objective="reg:squarederror", random_state=0)
    grid_search = GridSearchCV(
        xgb_model, param_grid, cv=5, scoring="neg_mean_squared_error"
    )
    grid_search.fit(X0_full, Y0_full)
    target_only_xgb = grid_search.best_estimator_
    xbg_pool = np.mean(abs(target_only_xgb.predict(X_test) - Y_test) ** 2)

    ## Kernel Ridge Regression
    alpha_lst = 0.1 / X0_full.shape[0] * (3.0 ** np.array(range(-2, 6)))
    param_grid = {"alpha": alpha_lst}
    target_only_krr = KernelRidge(kernel="rbf")
    grid_search = GridSearchCV(
        target_only_krr, param_grid, cv=5, scoring="neg_mean_squared_error"
    )
    grid_search.fit(X0_full, Y0_full)
    target_only_krr = grid_search.best_estimator_
    krr_pool = np.mean(abs(target_only_krr.predict(X_test) - Y_test) ** 2)

    ## Neural Network
    param_grid = {
        "hidden_layer_sizes": [(10,), (50,), (100,)],
        "alpha": [0.0001, 0.001, 0.01],
    }
    mlp = MLPRegressor(max_iter=1000, random_state=0)
    grid_search = GridSearchCV(mlp, param_grid, cv=5)
    grid_search.fit(X0_full, Y0_full)
    target_only_mlp = grid_search.best_estimator_
    nn_pool = np.mean(abs(target_only_mlp.predict(X_test) - Y_test) ** 2)

    # ML models with transfer learning with conditional quantile matching
    X_test = dat_test[:, 1:]
    Y_test = dat_test[:, 0]
    X_dat0 = dat0[:, 1:]
    Y0 = dat0[:, 0]

    # Fit the engression generative model on each source data
    eng_mod = []
    X_source_tensor = []
    for i in range(len(dat_source)):
        Y_tensor = torch.tensor(dat_source[i][:, 0].reshape(-1, 1), dtype=torch.float32)
        X_tensor = torch.tensor(dat_source[i][:, 1:], dtype=torch.float32)
        engressor = engression(
            X_tensor,
            Y_tensor,
            num_layer=2,
            hidden_dim=100,
            noise_dim=100,
            lr=0.0001,
            num_epochs=1000,
        )
        X_source_tensor.append(X_tensor)
        eng_mod.append(engressor)
    X_dat0_tensor = torch.tensor(X_dat0, dtype=torch.float32)
    X_source_tensor = torch.cat(X_source_tensor, dim=0)

    # Sample response variables from each source data based on the covariates in the target domain
    N_sam = 3000
    Y0_sam = []
    for i in range(len(eng_mod)):
        Y0_sam.append(
            eng_mod[i]
            .sample(X_dat0_tensor, sample_size=N_sam)
            .detach()
            .numpy()
            .reshape(-1, 1)
        )
    Y0_sam = np.concatenate(Y0_sam, axis=1)
    Y0_sam_arr = np.concatenate([np.ones([Y0_sam.shape[0], 1]), Y0_sam], axis=1)

    beta_sol = quantile_matching_estimate(
        np.repeat(Y0, N_sam),
        Y0_sam_arr,
        beta_init=None,
        stop_eps=1e-8,
        max_iter=1000,
        verbose=False,
    )

    Y_source_pred = []
    for i in range(len(eng_mod)):
        Y_source_pred.append(
            eng_mod[i]
            .predict(X_source_tensor, sample_size=100)
            .detach()
            .numpy()
            .reshape(-1, 1)
        )
    Y_source_pred = np.concatenate(Y_source_pred, axis=1)
    Y_source_pred = np.concatenate(
        [np.ones([Y_source_pred.shape[0], 1]), Y_source_pred], axis=1
    )
    Y_matched = np.dot(Y_source_pred, beta_sol)

    # Kernel mean matching for covariate shift correction
    X_source = X_source_tensor.detach().numpy()
    kmm_weights = kernel_mean_matching(X_test, X_source, kern="rbf", B=10)[:, 0]

    X_comb = np.concatenate([X_source, X_dat0], axis=0)
    Y_comb = np.concatenate([Y_matched, Y0], axis=0)
    weights = np.concatenate([kmm_weights, np.ones(X_dat0.shape[0])], axis=0)

    ## XGBoost
    param_grid = {
        "learning_rate": [0.001, 0.01, 0.1],
        "n_estimators": [10, 50, 100],
        "max_depth": [3, 5],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    xgb_model = XGBRegressor(objective="reg:squarederror", random_state=0)
    grid_search = GridSearchCV(
        xgb_model, param_grid, cv=5, scoring="neg_mean_squared_error"
    )
    grid_search.fit(X_comb, Y_comb, sample_weight=weights)
    target_only_xgb = grid_search.best_estimator_
    xbg_tlcqm = np.mean(abs(target_only_xgb.predict(X_test) - Y_test) ** 2)

    ## Kernel Ridge Regression
    alpha_lst = 0.1 / X_comb.shape[0] * (3.0 ** np.array(range(-2, 6)))
    param_grid = {"alpha": alpha_lst}
    target_only_krr = KernelRidge(kernel="rbf")
    grid_search = GridSearchCV(
        target_only_krr, param_grid, cv=5, scoring="neg_mean_squared_error"
    )
    grid_search.fit(X_comb, Y_comb, sample_weight=weights)
    target_only_krr = grid_search.best_estimator_
    krr_tlcqm = np.mean(abs(target_only_krr.predict(X_test) - Y_test) ** 2)

    ## Neural Network
    param_grid = {
        "hidden_layer_sizes": [(10,), (50,), (100,)],
        "alpha": [0.0001, 0.001, 0.01],
    }
    mlp = MLPRegressor(max_iter=1000, random_state=0)
    grid_search = GridSearchCV(mlp, param_grid, cv=5)
    grid_search.fit(X_comb, Y_comb)
    target_only_mlp = grid_search.best_estimator_
    nn_tlcqm = np.mean(abs(target_only_mlp.predict(X_test) - Y_test) ** 2)

    # Save results
    mse = np.array(
        [
            xbg_to,
            krr_to,
            nn_to,
            xbg_pool,
            krr_pool,
            nn_pool,
            xbg_tlcqm,
            krr_tlcqm,
            nn_tlcqm,
        ]
    )
    res_names = [
        "XGBoost_Target_Only",
        "KRR_Target_Only",
        "NN_Target_Only",
        "XGBoost_Oracle",
        "KRR_Oracle",
        "NN_Oracle",
        "XGBoost_TLCQM",
        "KRR_TLCQM",
        "NN_TLCQM",
    ]
    res_df = pd.DataFrame({"Method": res_names, "MSE": mse})
    res_df["target_size"] = n_0
    res_full = pd.concat([res_full, res_df], axis=0)

res_full.to_csv("./Results/Apartment_" + str(job_id) + ".csv", index=False)
