#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang
Last Editing: Jan 2, 2026

Description: Application to the UCI Apartment for rent data.
This file contains code for transfer learning models to be compared with.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from utils import sim_data
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from itertools import combinations
import torch
import torch.nn as nn
import torch.optim as optim
import sys

job_id = int(sys.argv[1])
print(job_id)

#=======================================================================================#

def fit_krr(X, Y, alpha_grid=None):
    if alpha_grid is None:
        alpha_grid = 0.1 / X.shape[0] * (3.0 ** np.arange(-3, 7))
    param_grid = {'alpha': alpha_grid}
    krr = KernelRidge(kernel="rbf")
    grid_search = GridSearchCV(krr, param_grid, cv=5, scoring="neg_mean_squared_error")
    grid_search.fit(X, Y)
    return grid_search.best_estimator_
def rkhs_norm(f1, f2, X):
    # Use L2 norm of predictions as a proxy for RKHS norm
    return np.linalg.norm(f1.predict(X) - f2.predict(X))

#=======================================================================================#

# Example TargetCNN definition for d=5
class TargetCNN(nn.Module):
    def __init__(self, d=5):
        super(TargetCNN, self).__init__()
        self.fc1 = nn.Linear(d, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 8)
        self.predict = nn.Linear(8, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        inter_x1 = self.relu(x)
        x = self.fc2(inter_x1)
        inter_x2 = self.relu(x)
        x = self.fc3(inter_x2)
        inter_x3 = self.relu(x)
        result = self.predict(inter_x3)
        target_list = [inter_x1, inter_x2, inter_x3]
        return target_list, result

def to_tensor(X, Y):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).reshape(-1, 1)
    return X_tensor, Y_tensor

def rbf_kernel(X, Y, gamma=0.4):
    # X: (n_samples_X, n_features)
    # Y: (n_samples_Y, n_features)
    X = X if X.ndim == 2 else X.view(X.size(0), -1)
    Y = Y if Y.ndim == 2 else Y.view(Y.size(0), -1)
    XX = torch.sum(X ** 2, 1).view(-1, 1)
    YY = torch.sum(Y ** 2, 1).view(1, -1)
    distances = XX + YY - 2 * torch.mm(X, Y.t())
    K = torch.exp(-gamma * distances)
    return K

def MLcon_kernel(source_list, source_pred, target_list, target_y, lamda=1.0):
    # Use only the first layer's features for simplicity
    X_p = source_list[0]  # (n_source, n_features)
    Y_p = source_pred     # (n_source, 1)
    X_q = target_list[0]  # (n_target, n_features)
    Y_q = target_y        # (n_target, 1)

    np_ = X_p.shape[0]
    nq_ = X_q.shape[0]
    I1 = torch.eye(np_, device=X_p.device)
    I2 = torch.eye(nq_, device=X_q.device)

    Kxpxp = rbf_kernel(X_p, X_p)
    Kxqxq = rbf_kernel(X_q, X_q)
    Kxqxp = rbf_kernel(X_q, X_p)
    Kypyq = rbf_kernel(Y_p, Y_q)
    Kyqyq = rbf_kernel(Y_q, Y_q)
    Kypyp = rbf_kernel(Y_p, Y_p)

    a = torch.mm(torch.inverse(Kxpxp + np_ * lamda * I1), Kypyp)
    b = torch.mm(a, torch.inverse(Kxpxp + np_ * lamda * I1))
    c = torch.mm(b, Kxpxp)
    out1 = torch.trace(c)

    a1 = torch.mm(torch.inverse(Kxqxq + nq_ * lamda * I2), Kyqyq)
    b1 = torch.mm(a1, torch.inverse(Kxqxq + nq_ * lamda * I2))
    c1 = torch.mm(b1, Kxqxq)
    out2 = torch.trace(c1)

    a2 = torch.mm(torch.inverse(Kxpxp + np_ * lamda * I1), Kypyq)
    b2 = torch.mm(a2, torch.inverse(Kxqxq + nq_ * lamda * I2))
    c2 = torch.mm(b2, Kxqxp)
    out3 = torch.trace(c2)

    out = out1 + out2 - 2 * out3
    return out

#=======================================================================================#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, domain_dim, feature_dim=2):
        super().__init__()
        self.domain_embed = nn.Embedding(domain_dim, 8)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + 8, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, feature_dim)
        )

    def forward(self, x, domain_id):
        domain_vec = self.domain_embed(domain_id)
        x_cat = torch.cat([x, domain_vec], dim=1)
        return self.mlp(x_cat)

def psp_loss(features, labels):
    dist_matrix = torch.cdist(features, features, p=2)
    label_matrix = torch.abs(labels.unsqueeze(0) - labels.unsqueeze(1))
    return nn.functional.mse_loss(dist_matrix, label_matrix)

class LinearRegressor(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.linear = nn.Linear(feature_dim, 1)

    def forward(self, features):
        return self.linear(features).squeeze(-1)

def train_feature_extractor(F, X, Y, domain_ids, epochs=1000, lr=1e-3):
    F = F.to(device)
    optimizer = optim.Adam(F.parameters(), lr=lr)
    X = torch.tensor(X, dtype=torch.float32).to(device)
    Y = torch.tensor(Y, dtype=torch.float32).to(device)
    domain_ids = torch.tensor(domain_ids, dtype=torch.long).to(device)
    for epoch in range(epochs):
        F.train()
        features = F(X, domain_ids)
        loss = psp_loss(features, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}, PSP Loss: {loss.item():.4f}")
    return F

def train_regressor(F, R, X, Y, domain_ids, epochs=1000, lr=1e-3):
    F = F.to(device)
    R = R.to(device)
    optimizer = optim.Adam(R.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    X = torch.tensor(X, dtype=torch.float32).to(device)
    Y = torch.tensor(Y, dtype=torch.float32).to(device)
    domain_ids = torch.tensor(domain_ids, dtype=torch.long).to(device)
    for epoch in range(epochs):
        F.eval()
        with torch.no_grad():
            features = F(X, domain_ids)
        preds = R(features)
        loss = loss_fn(preds, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}, Regressor Loss: {loss.item():.4f}")
    return R

#=======================================================================================#


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

source_domain = ["CA", "TX", "VA", "NC"]
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
    dat0 = data_sub.sample(n=n_0, random_state=job_id)
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
    
    # Prepare data
    X_source = [dat[:, 1:] for dat in dat_source]
    Y_source = [dat[:, 0] for dat in dat_source]
    X0 = dat0[:, 1:]
    Y0 = dat0[:, 0]
    X0_test = dat_test[:, 1:]
    Y0_test = dat_test[:, 0]

    # TKRR
    m = len(X_source)
    # --- Step 1: Split target data into T1 and T2 ---
    X0_T1, X0_T2, Y0_T1, Y0_T2 = train_test_split(X0, Y0, test_size=0.5, random_state=0)
    # Further split T2 into T21 and T22 for aggregation
    X0_T21, X0_T22, Y0_T21, Y0_T22 = train_test_split(X0_T2, Y0_T2, test_size=0.5, random_state=0)
    # --- Step 2: Fit KRR on each source and T1 ---
    fb0 = fit_krr(X0_T1, Y0_T1)  # Target model on T1
    fbk_list = [fit_krr(Xk, Yk) for Xk, Yk in zip(X_source, Y_source)]  # Source models
    # --- Step 3: Compute contrast functions and RKHS norms ---
    norms = [rkhs_norm(fbk, fb0, X0_T1) for fbk in fbk_list]
    ranks = np.argsort(norms)  # Indices of sources sorted by similarity
    # --- Step 4: Build candidate models using increasing numbers of sources ---
    candidate_models = [fb0]  # fb0 corresponds to Ab0 = ∅
    for ell in range(1, m+1):
        selected_indices = ranks[:ell]
        # Combine selected sources and T1
        X_comb = np.concatenate([X_source[i] for i in selected_indices] + [X0_T1], axis=0)
        Y_comb = np.concatenate([Y_source[i] for i in selected_indices] + [Y0_T1], axis=0)
        # Transferring step
        comb_krr = fit_krr(X_comb, Y_comb)
        # Debiasing step
        Y0_pred = comb_krr.predict(X0_T1)
        Y_resi = Y0_T1 - Y0_pred
        resi_krr = fit_krr(X0_T1, Y_resi)
        # Final model: sum of transferring and debiasing
        class CombinedModel:
            def __init__(self, m1, m2):
                self.m1 = m1
                self.m2 = m2
            def predict(self, X):
                return self.m1.predict(X) + self.m2.predict(X)
        fb_ell = CombinedModel(comb_krr, resi_krr)
        candidate_models.append(fb_ell)
    # --- Step 5: Hyper-sparse aggregation (convex combination of at most two models) ---
    # Evaluate risk on T21 for each candidate
    risks = [np.mean((model.predict(X0_T21) - Y0_T21)**2) for model in candidate_models]
    best_idx = np.argmin(risks)
    # Find best convex combination of two models
    min_risk = risks[best_idx]
    best_combo = (best_idx, None, 1.0)
    for i, j in combinations(range(len(candidate_models)), 2):
        # Find optimal t in [0,1] for convex combination
        preds_i = candidate_models[i].predict(X0_T21)
        preds_j = candidate_models[j].predict(X0_T21)
        t_vals = np.linspace(0, 1, 101)
        for t in t_vals:
            preds = t * preds_i + (1-t) * preds_j
            risk = np.mean((preds - Y0_T21)**2)
            if risk < min_risk:
                min_risk = risk
                best_combo = (i, j, t)
    # Build final aggregated model
    i, j, t = best_combo
    class AggregatedModel:
        def __init__(self, m1, m2, t):
            self.m1 = m1
            self.m2 = m2
            self.t = t
        def predict(self, X):
            if self.m2 is None:
                return self.m1.predict(X)
            return self.t * self.m1.predict(X) + (1-self.t) * self.m2.predict(X)
    if j is not None:
        fba = AggregatedModel(candidate_models[i], candidate_models[j], t)
    else:
        fba = AggregatedModel(candidate_models[i], None, t)
    # --- Step 6: Evaluate on test set ---
    Y0_pred_new = fba.predict(X0_test)
    tkrr_mse = np.mean((Y0_pred_new - Y0_test) ** 2)


    # CDAR
    # Prepare data
    X_source = [dat[:, 1:] for dat in dat_source]
    Y_source = [dat[:, 0] for dat in dat_source]
    X0_train = dat0[:, 1:]
    Y0_train = dat0[:, 0]
    X0_test = dat_test[:, 1:]
    Y0_test = dat_test[:, 0]

    # Convert to tensors
    X_source_tensor, Y_source_tensor = to_tensor(X_source[0], Y_source[0])  # Use first source domain
    X0_train_tensor, Y0_train_tensor = to_tensor(X0_train, Y0_train)
    X0_test_tensor, Y0_test_tensor = to_tensor(X0_test, Y0_test)

    # Initialize model
    model = TargetCNN(d=d)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()

    # CDAR hyperparameters
    Lambda = 1.0
    Beta = 1.0
    num_epochs = 100

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        # Forward pass for source and target
        source_list, source_pred = model(X_source_tensor)
        target_list, target_pred = model(X0_train_tensor)

        # Compute CEOD loss (replace with your actual implementation)
        CEOD_loss = MLcon_kernel(
        source_list, source_pred, target_list, Y0_train_tensor
        )

        # Hybrid loss
        loss = Lambda * criterion(target_pred, Y0_train_tensor) + Beta * CEOD_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        _, y_pred = model(X0_test_tensor)
        y_pred = y_pred.numpy().flatten()
        cdar_mse = np.mean((Y0_test - y_pred) ** 2)


    # DARC
    # Multi-source domains
    num_source_domains = len(dat_source)
    # Stack all source domains
    X_source = np.vstack([dat[:, 1:] for dat in dat_source])
    Y_source = np.concatenate([dat[:, 0] for dat in dat_source])
    domain_ids_source = np.concatenate([
        np.full(len(dat_source[i]), i, dtype=int) for i in range(num_source_domains)
    ])
    # Target domain
    X_target = dat0[:, 1:]
    Y_target = dat0[:, 0]
    domain_ids_target = np.full(len(X_target), num_source_domains, dtype=int)  # Target domain ID
    # Combine for training
    X_train = np.vstack([X_source, X_target])
    Y_train = np.concatenate([Y_source, Y_target])
    domain_ids_train = np.concatenate([domain_ids_source, domain_ids_target])
    # Train DARC feature extractor
    F = FeatureExtractor(input_dim=d, domain_dim=num_source_domains+1, feature_dim=2)
    F = train_feature_extractor(F, X_train, Y_train, domain_ids_train, epochs=1000, lr=1e-3)
    # Train linear regressor on constructed space
    R = LinearRegressor(feature_dim=2)
    R = train_regressor(F, R, X_train, Y_train, domain_ids_train, epochs=1000, lr=1e-3)
    # Evaluate on test set (target domain)
    X_test = dat_test[:, 1:]
    Y_test = dat_test[:, 0]
    domain_ids_test = np.full(len(X_test), num_source_domains, dtype=int)
    X_test_torch = torch.tensor(X_test, dtype=torch.float32).to(device)
    domain_ids_test_torch = torch.tensor(domain_ids_test, dtype=torch.long).to(device)
    with torch.no_grad():
        features_test = F(X_test_torch, domain_ids_test_torch)
        preds = R(features_test).cpu().numpy()
    darc_mse = np.mean((preds - Y_test)**2)


    # Save results
    mse = np.array([tkrr_mse, cdar_mse, darc_mse])
    res_names = ['TKRR', 'CDAR', 'DARC']
    res_df = pd.DataFrame({'Method': res_names, 'MSE': mse})
    res_df['Sample_size'] = n_s
    res_full = pd.concat([res_full, res_df], axis=0)

res_full.to_csv('./Results/Apartment_'+str(job_id)+'_Compare.csv', index=False)