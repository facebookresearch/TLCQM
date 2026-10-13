# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
@author: Yikun Zhang
Last Editing: Apr 1, 2026

Description: Application to the UCI Apartment for rent data.
It contains XGBoost, kernel ridge regression, and neural network models
applied to the source-only, naive pooling, and mixup augmented data.
"""

import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor

job_id = int(sys.argv[1])
print(job_id)

# =======================================================================================#

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


def fit_eval_xgb(X_train, y_train, X_test, y_test, sample_weight=None):
    param_grid = {
        "learning_rate": [0.001, 0.01, 0.1],
        "n_estimators": [10, 50, 100],
        "max_depth": [3, 5],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    model = XGBRegressor(objective="reg:squarederror", random_state=0)
    gs = GridSearchCV(model, param_grid, cv=5, scoring="neg_mean_squared_error")
    gs.fit(X_train, y_train, sample_weight=sample_weight)
    pred = gs.best_estimator_.predict(X_test)
    return np.mean((pred - y_test) ** 2)


def fit_eval_krr(X_train, y_train, X_test, y_test, sample_weight=None):
    alpha_lst = 0.1 / X_train.shape[0] * (3.0 ** np.arange(-2, 6))
    gs = GridSearchCV(
        KernelRidge(kernel="rbf"),
        {"alpha": alpha_lst},
        cv=5,
        scoring="neg_mean_squared_error",
    )
    gs.fit(X_train, y_train, sample_weight=sample_weight)
    pred = gs.best_estimator_.predict(X_test)
    return np.mean((pred - y_test) ** 2)


def fit_eval_nn(X_train, y_train, X_test, y_test):
    # MLPRegressor may not support sample_weight depending on sklearn version.
    param_grid = {
        "hidden_layer_sizes": [(10,), (50,), (100,)],
        "alpha": [0.0001, 0.001, 0.01],
    }
    gs = GridSearchCV(
        MLPRegressor(max_iter=1000, random_state=0),
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
    )
    gs.fit(X_train, y_train)
    pred = gs.best_estimator_.predict(X_test)
    return np.mean((pred - y_test) ** 2)


def eval_all_models(X_train, y_train, X_test, y_test, sample_weight=None):
    out = {}
    out["XGBoost"] = fit_eval_xgb(X_train, y_train, X_test, y_test, sample_weight)
    out["KRR"] = fit_eval_krr(X_train, y_train, X_test, y_test, sample_weight)
    out["NN"] = fit_eval_nn(X_train, y_train, X_test, y_test)
    return out


def build_feature_df(df):

    X = pd.get_dummies(
        df[
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
            ]
        ],
        columns=["has_photo"],
        dtype=int,
    )

    y = df["price"].values

    return X.values, y


def build_pool_feature_df(train_df, test_df):
    train_n = train_df.shape[0]
    both = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    both = pd.get_dummies(both, columns=["has_photo", "state"], dtype=int)

    train_out = both.iloc[:train_n, :].copy()
    test_out = both.iloc[train_n:, :].copy()

    y_train = train_out["price"].values
    y_test = test_out["price"].values

    X_train = train_out.drop(columns=["price"]).values
    X_test = test_out.drop(columns=["price"]).values

    return X_train, y_train, X_test, y_test

# Read the data and preprocessing
data_raw = pd.read_csv("data/apartments_for_rent_classified_100K.csv", 
                       encoding="latin1", engine="python", on_bad_lines="skip",
                       sep=";")

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
    "price"]
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

data_ap["price"] = np.log(data_ap["price"])

source_domain = ["IL", "OH", "WA"]
target_domain = "FL"

res_full = []

for n_0 in [100, 200, 300, 500]:
    np.random.seed(job_id)

    # Prepare source data
    dat_source = []
    source_named = {}

    for sname in source_domain:
        df_s = data_ap.loc[data_ap["state"] == sname, :].copy()
        X_s, Y_s = build_feature_df(df_s)
        arr_s = np.column_stack([Y_s, X_s])
        dat_source.append(arr_s)
        source_named[sname] = arr_s

    # -------------------------
    # Prepare target train/test
    # -------------------------
    df_target_all = data_ap.loc[data_ap["state"] == target_domain, :].copy()
    df_target_train = df_target_all.sample(n=n_0, random_state=job_id)
    df_target_test = df_target_all.drop(df_target_train.index).copy()

    X0, Y0 = build_feature_df(df_target_train)
    X_test, Y_test = build_feature_df(df_target_test)

    # ============================================================
    # Source-only (each source separately)
    for sname, arr_s in source_named.items():
        Xs = arr_s[:, 1:]
        Ys = arr_s[:, 0]
        res_s = eval_all_models(Xs, Ys, X_test, Y_test)
        for model_name, mse in res_s.items():
            res_full.append({
                "Method": f"{model_name}_Source_{sname}",
                "MSE": mse,
                "target_size": n_0
            })

    # ============================================================
    # 3) Naive pooling (source + target train)
    #    Add state dummies only for pooled baseline
    # ============================================================
    pool_parts = []
    for sname in source_domain:
        pool_parts.append(
            data_ap.loc[
                data_ap["state"] == sname,
                [
                    "price", "state", "bathrooms", "bedrooms", "has_photo",
                    "square_feet", "Parking", "Storage", "Gym", "Pool", "Cats", "Dogs"
                ]
            ].copy()
        )
    pool_parts.append(
        df_target_train[
            [
                "price", "state", "bathrooms", "bedrooms", "has_photo",
                "square_feet", "Parking", "Storage", "Gym", "Pool", "Cats", "Dogs"
            ]
        ].copy()
    )

    pool_train_df = pd.concat(pool_parts, axis=0)
    pool_test_df = df_target_test[
        [
            "price", "state", "bathrooms", "bedrooms", "has_photo",
            "square_feet", "Parking", "Storage", "Gym", "Pool", "Cats", "Dogs"
        ]
    ].copy()

    X_pool, Y_pool, X_pool_test, Y_pool_test = build_pool_feature_df(pool_train_df, pool_test_df)

    res_pool = eval_all_models(X_pool, Y_pool, X_pool_test, Y_pool_test)
    for model_name, mse in res_pool.items():
        res_full.append({
            "Method": f"{model_name}_Naive_Pool",
            "MSE": mse,
            "target_size": n_0
        })

    # ============================================================
    # 4) Target-only Mixup
    #    Save both all-alpha results and best-alpha summary
    # ============================================================
    alpha_grid = [0.2, 0.4, 0.5, 0.6, 0.8]
    mixup_best = {"XGBoost": (None, np.inf), "KRR": (None, np.inf), "NN": (None, np.inf)}

    for alpha in alpha_grid:
        X_mix, Y_mix = make_mixup_regression(X0, Y0, alpha=alpha, n_aug=5 * len(Y0), random_state=job_id)
        X0_aug = np.vstack([X0, X_mix])
        Y0_aug = np.concatenate([Y0, Y_mix])

        res_mix = eval_all_models(X0_aug, Y0_aug, X_test, Y_test)

        for model_name, mse in res_mix.items():
            res_full.append({
                "Method": f"{model_name}_Mixup",
                "MSE": mse,
                "target_size": n_0,
                "alpha": alpha
            })
            if mse < mixup_best[model_name][1]:
                mixup_best[model_name] = (alpha, mse)

    # store best mixup per learner
    for model_name, (best_alpha, best_mse) in mixup_best.items():
        res_full.append({
            "Method": f"{model_name}_Mixup_Best",
            "MSE": best_mse,
            "target_size": n_0,
            "alpha": best_alpha
        })

res_full = pd.DataFrame(res_full)
res_full.to_csv("./Results/Apartment_" + str(job_id) + "_baselines_extended.csv", index=False)

