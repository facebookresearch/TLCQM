# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
@author: Yikun Zhang
Last Editing: Mar 25, 2026

Description: Simulation on data with both concept and covariate shifts.
This file contains code for transfer learning models to be compared with.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.kernel_ridge import KernelRidge
from itertools import combinations
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from utils import sim_data

job_id = int(sys.argv[1])
print(job_id)


# ============================================================
# Existing utilities
# ============================================================

def fit_krr(X, Y, alpha_grid=None):
    if alpha_grid is None:
        alpha_grid = 0.1 / max(1, X.shape[0]) * (3.0 ** np.arange(-3, 7))
    param_grid = {'alpha': alpha_grid}
    krr = KernelRidge(kernel="rbf")
    grid_search = GridSearchCV(
        krr,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error"
    )
    grid_search.fit(X, Y)
    return grid_search.best_estimator_


def rkhs_norm(f1, f2, X):
    return np.linalg.norm(f1.predict(X) - f2.predict(X))


def to_tensor(X, Y=None, device="cpu"):
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    if Y is None:
        return X_tensor
    Y_tensor = torch.tensor(Y, dtype=torch.float32, device=device).reshape(-1, 1)
    return X_tensor, Y_tensor


# ============================================================
# Shared deep modules
# ============================================================

class FeatureMLP(nn.Module):
    def __init__(self, input_dim, hidden=(64, 32), feat_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], feat_dim),
        )

    def forward(self, x):
        return self.net(x)


class RegressorHead(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(feat_dim, 1),
        )

    def forward(self, z):
        return self.net(z)


class DomainDiscriminator(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


class CurriculumManager(nn.Module):
    """
    CMSS-style per-sample weighting network.
    Outputs positive weights and rescales them so their batch mean is 1.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )

    def forward(self, x):
        w = self.net(x).squeeze(-1) + 1e-6
        return w / w.mean().clamp_min(1e-6)


def clone_module(module):
    import copy
    return copy.deepcopy(module)


# ============================================================
# von Neumann conditional divergence (MDD-style) adapted to fully-labeled regression simulation
# ============================================================

def _cov_spd(x, eps=1e-4):
    """
    Empirical covariance with ridge stabilization.
    x: (n, p)
    returns: (p, p) SPD matrix
    """
    x = x - x.mean(dim=0, keepdim=True)
    n, p = x.shape
    cov = (x.T @ x) / max(n - 1, 1)
    return cov + eps * torch.eye(p, device=x.device, dtype=x.dtype)


def _matrix_log_spd(A, eps=1e-8):
    eigvals, eigvecs = torch.linalg.eigh(A)
    eigvals = torch.clamp(eigvals, min=eps)
    return eigvecs @ torch.diag(torch.log(eigvals)) @ eigvecs.T


def dvn_divergence(Sigma, Rho):
    """
    von Neumann divergence:
        tr(S log S - S log R - S + R)
    """
    logS = _matrix_log_spd(Sigma)
    logR = _matrix_log_spd(Rho)
    return torch.trace(Sigma @ logS - Sigma @ logR - Sigma + Rho)


def jvn_divergence(Sigma, Rho):
    """
    Jeffreys-von-Neumann divergence:
        0.5 * (DvN(S||R) + DvN(R||S))
    Equivalent to the symmetric form used in the paper.
    """
    return 0.5 * (dvn_divergence(Sigma, Rho) + dvn_divergence(Rho, Sigma))


def joint_covariance(x, y, eps=1e-4):
    """
    x: (n, p)
    y: (n, 1)
    joint covariance of [x, y].
    """
    xy = torch.cat([x, y], dim=1)
    return _cov_spd(xy, eps=eps)


def vnc_divergence_between_domains(x1, y1, x2, y2, eps=1e-4):
    """
    Symmetric von Neumann conditional divergence, Eq. (4) in Paper 1:
      D(P1(y|x) : P2(y|x))
      = 1/2 [ DvN(sig_xy || rho_xy) - DvN(sig_x || rho_x)
            + DvN(rho_xy || sig_xy) - DvN(rho_x || sig_x) ]
    """
    sig_xy = joint_covariance(x1, y1, eps)
    rho_xy = joint_covariance(x2, y2, eps)
    sig_x = _cov_spd(x1, eps)
    rho_x = _cov_spd(x2, eps)

    d12 = dvn_divergence(sig_xy, rho_xy) - dvn_divergence(sig_x, rho_x)
    d21 = dvn_divergence(rho_xy, sig_xy) - dvn_divergence(rho_x, sig_x)
    return 0.5 * (d12 + d21)


def prediction_jvn_loss(x, y_true, y_pred, eps=1e-4):
    """
    The paper notes that when the input x is shared, the divergence reduces
    to JvN between covariance matrices of (x, y_true) and (x, y_pred).
    We use this as the source supervised fit term.
    """
    sig = joint_covariance(x, y_true, eps)
    rho = joint_covariance(x, y_pred, eps)
    return torch.sqrt(torch.clamp(jvn_divergence(sig, rho), min=1e-12))


def fit_vncd_regression(
    X_sources,
    Y_sources,
    X_target,
    Y_target,
    X_test,
    num_epochs=300,
    lr=1e-3,
    feat_dim=16,
    hidden=(64, 32),
    adv_steps=1,
    lambda_disc=0.5,
    lambda_tgt=1.0,
    device=None,
    seed=0,
):
    """
    Adaptation of Paper 1 to the user's regression setup.

    We use:
      - feature extractor f_theta
      - predictor h
      - adversarial predictor h_adv
    and optimize a practical version of Eq. (8):
      source supervised JvN fit
      + target supervised MSE (since your target labels are available)
      + discrepancy term that encourages source/target predictive geometry to align.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    np.random.seed(seed)

    d = X_target.shape[1]
    m = len(X_sources)

    F = FeatureMLP(d, hidden=hidden, feat_dim=feat_dim).to(device)
    H = RegressorHead(feat_dim).to(device)
    H_adv = RegressorHead(feat_dim).to(device)

    opt_main = optim.Adam(list(F.parameters()) + list(H.parameters()), lr=lr)
    opt_adv = optim.Adam(H_adv.parameters(), lr=lr)

    Xs = [to_tensor(X, device=device) for X in X_sources]
    Ys = [to_tensor(None if False else np.zeros((1,1)), device=device) if False else torch.tensor(y, dtype=torch.float32, device=device).reshape(-1, 1) for y in Y_sources]
    Xt, Yt = to_tensor(X_target, Y_target, device=device)
    Xtest = to_tensor(X_test, device=device)

    w = torch.ones(m, device=device) / m

    for _ in range(num_epochs):
        # update adversarial predictor
        for _ in range(adv_steps):
            with torch.no_grad():
                zt = F(Xt)
                zs = [F(x) for x in Xs]

            pred_t = H(zt).detach()
            pred_t_adv = H_adv(zt)
            disc_adv = prediction_jvn_loss(zt, pred_t, pred_t_adv)

            source_disc = 0.0
            for j in range(m):
                pred_s = H(zs[j]).detach()
                pred_s_adv = H_adv(zs[j])
                source_disc = source_disc + w[j] * prediction_jvn_loss(zs[j], pred_s, pred_s_adv)

            # maximize target-source discrepancy => minimize negative
            adv_loss = -(disc_adv - source_disc)
            opt_adv.zero_grad()
            adv_loss.backward()
            opt_adv.step()

        # update source weights based on closeness to target (smaller divergence => larger weight)
        with torch.no_grad():
            zt = F(Xt)
            domain_scores = []
            for j in range(m):
                zj = F(Xs[j])
                score = vnc_divergence_between_domains(zj, Ys[j], zt, Yt)
                domain_scores.append(score)
            domain_scores = torch.stack(domain_scores)
            w = torch.softmax(-domain_scores, dim=0)

        # update F and H
        zt = F(Xt)
        pred_t = H(zt)
        tgt_loss = nn.functional.mse_loss(pred_t, Yt)

        src_fit = 0.0
        for j in range(m):
            zj = F(Xs[j])
            pred_s = H(zj)
            src_fit = src_fit + w[j] * prediction_jvn_loss(zj, Ys[j], pred_s)

        pred_t_adv = H_adv(zt)
        disc_main = prediction_jvn_loss(zt, pred_t, pred_t_adv)

        source_disc = 0.0
        for j in range(m):
            zj = F(Xs[j])
            pred_s = H(zj)
            pred_s_adv = H_adv(zj)
            source_disc = source_disc + w[j] * prediction_jvn_loss(zj, pred_s, pred_s_adv)

        discrepancy = disc_main - source_disc

        loss = src_fit + lambda_tgt * tgt_loss + lambda_disc * discrepancy

        opt_main.zero_grad()
        loss.backward()
        opt_main.step()

    F.eval()
    H.eval()
    with torch.no_grad():
        yhat = H(F(Xtest)).cpu().numpy().reshape(-1)
    return yhat


# ============================================================
# M3SDA / M3SDA- adapted to regression
# ============================================================

def central_moment(x, k):
    return ((x - x.mean(dim=0, keepdim=True)) ** k).mean(dim=0)


def moment_distance(x1, x2, max_order=5):
    """
    Practical central-moment distance for latent features.
    This is the regression analogue of the moment matching component in Paper 2.
    """
    loss = torch.norm(x1.mean(dim=0) - x2.mean(dim=0), p=2)
    for k in range(2, max_order + 1):
        loss = loss + torch.norm(central_moment(x1, k) - central_moment(x2, k), p=2)
    return loss


class DomainSpecificRegressors(nn.Module):
    def __init__(self, n_domains, feat_dim):
        super().__init__()
        self.heads = nn.ModuleList([RegressorHead(feat_dim) for _ in range(n_domains)])

    def forward(self, z):
        return [head(z) for head in self.heads]


def fit_m3sda_regression(
    X_sources,
    Y_sources,
    X_target,
    Y_target,
    X_test,
    num_epochs=300,
    lr=1e-3,
    feat_dim=16,
    hidden=(64, 32),
    lambda_moment=0.5,
    discrepancy_steps=3,
    device=None,
    seed=0,
):
    """
    Paper 2 is a classification MSDA method based on:
      - matching moments target<->source and source<->source
      - optional M3SDA-beta discrepancy training with two heads per source
    Here we use the M3SDA-beta-style alternating procedure, but with regression heads.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    np.random.seed(seed)

    d = X_target.shape[1]
    m = len(X_sources)

    F = FeatureMLP(d, hidden=hidden, feat_dim=feat_dim).to(device)
    H1 = DomainSpecificRegressors(m, feat_dim).to(device)
    H2 = DomainSpecificRegressors(m, feat_dim).to(device)

    opt_feat = optim.Adam(F.parameters(), lr=lr)
    opt_heads = optim.Adam(list(H1.parameters()) + list(H2.parameters()), lr=lr)

    Xs = [to_tensor(X, device=device) for X in X_sources]
    Ys = [torch.tensor(y, dtype=torch.float32, device=device).reshape(-1, 1) for y in Y_sources]
    Xt, Yt = to_tensor(X_target, Y_target, device=device)
    Xtest = to_tensor(X_test, device=device)

    # target closeness weights, derived from source-only risk on target labels
    source_only_mse = []
    for j in range(m):
        model = fit_krr(X_sources[j], Y_sources[j])
        source_only_mse.append(np.mean((model.predict(X_target) - Y_target) ** 2))
    source_w = np.exp(-np.array(source_only_mse))
    source_w = source_w / source_w.sum()
    source_w = torch.tensor(source_w, dtype=torch.float32, device=device)

    for _ in range(num_epochs):
        # ----------------------------------------------------
        # Step A: supervised regression + moment matching
        # ----------------------------------------------------
        zt = F(Xt)
        zs = [F(x) for x in Xs]
        pred_t_1 = [H1.heads[j](zt) for j in range(m)]
        pred_t_2 = [H2.heads[j](zt) for j in range(m)]

        sup_loss = 0.0
        for j in range(m):
            ps1 = H1.heads[j](zs[j])
            ps2 = H2.heads[j](zs[j])
            sup_loss = sup_loss + source_w[j] * (
                nn.functional.mse_loss(ps1, Ys[j]) +
                nn.functional.mse_loss(ps2, Ys[j])
            )

        # Use target labels because your simulation provides them.
        tgt_loss = 0.0
        for j in range(m):
            tgt_loss = tgt_loss + source_w[j] * (
                nn.functional.mse_loss(pred_t_1[j], Yt) +
                nn.functional.mse_loss(pred_t_2[j], Yt)
            )

        mm_loss = 0.0
        for j in range(m):
            mm_loss = mm_loss + moment_distance(zs[j], zt)
        for i in range(m):
            for j in range(i + 1, m):
                mm_loss = mm_loss + moment_distance(zs[i], zs[j])

        loss_a = sup_loss + 0.5 * tgt_loss + lambda_moment * mm_loss
        opt_feat.zero_grad()
        opt_heads.zero_grad()
        loss_a.backward()
        opt_feat.step()
        opt_heads.step()

        # ----------------------------------------------------
        # Step B: fix F, maximize target discrepancy between paired heads
        # ----------------------------------------------------
        for p in F.parameters():
            p.requires_grad = False
        for p in H1.parameters():
            p.requires_grad = True
        for p in H2.parameters():
            p.requires_grad = True

        zt = F(Xt).detach()
        zs = [F(x).detach() for x in Xs]
        sup_loss_b = 0.0
        disc = 0.0
        for j in range(m):
            ps1 = H1.heads[j](zs[j])
            ps2 = H2.heads[j](zs[j])
            sup_loss_b = sup_loss_b + source_w[j] * (
                nn.functional.mse_loss(ps1, Ys[j]) +
                nn.functional.mse_loss(ps2, Ys[j])
            )
            pt1 = H1.heads[j](zt)
            pt2 = H2.heads[j](zt)
            disc = disc + source_w[j] * torch.mean(torch.abs(pt1 - pt2))

        loss_b = sup_loss_b - disc
        opt_heads.zero_grad()
        loss_b.backward()
        opt_heads.step()

        # ----------------------------------------------------
        # Step C: fix heads, minimize target discrepancy by updating F
        # ----------------------------------------------------
        for p in F.parameters():
            p.requires_grad = True
        for p in H1.parameters():
            p.requires_grad = False
        for p in H2.parameters():
            p.requires_grad = False

        for _ in range(discrepancy_steps):
            zt = F(Xt)
            disc_c = 0.0
            mm_loss_c = 0.0
            zs = [F(x) for x in Xs]
            for j in range(m):
                pt1 = H1.heads[j](zt)
                pt2 = H2.heads[j](zt)
                disc_c = disc_c + source_w[j] * torch.mean(torch.abs(pt1 - pt2))
                mm_loss_c = mm_loss_c + moment_distance(zs[j], zt)
            loss_c = disc_c + 0.1 * mm_loss_c
            opt_feat.zero_grad()
            loss_c.backward()
            opt_feat.step()

        for p in H1.parameters():
            p.requires_grad = True
        for p in H2.parameters():
            p.requires_grad = True

    F.eval()
    H1.eval()
    H2.eval()
    with torch.no_grad():
        ztest = F(Xtest)
        preds = []
        for j in range(m):
            pj = 0.5 * (H1.heads[j](ztest) + H2.heads[j](ztest))
            preds.append(pj.reshape(-1))
        preds = torch.stack(preds, dim=0)
        yhat = (source_w[:, None] * preds).sum(dim=0).cpu().numpy()
    return yhat


# ============================================================
# CMSS adapted to regression
# ============================================================

def bce_logits(logits, targets, weights=None):
    losses = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    if weights is not None:
        losses = losses * weights
    return losses.mean()


def fit_cmss_regression(
    X_sources,
    Y_sources,
    X_target,
    Y_target,
    X_test,
    num_epochs=300,
    lr=1e-3,
    feat_dim=16,
    hidden=(64, 32),
    lambda_adv=0.5,
    lambda_tgt=1.0,
    gamma=10.0,
    device=None,
    seed=0,
):
    """
    Paper 3 adapts DANN with a curriculum manager that outputs per-sample source
    weights and is trained to increase discriminator error.
    In your regression setup we retain the same idea, replacing source
    classification with source regression + target regression.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    np.random.seed(seed)

    d = X_target.shape[1]
    Xs = np.vstack(X_sources)
    Ys = np.concatenate(Y_sources)

    Xs_t, Ys_t = to_tensor(Xs, Ys, device=device)
    Xt_t, Yt_t = to_tensor(X_target, Y_target, device=device)
    Xtest_t = to_tensor(X_test, device=device)

    F = FeatureMLP(d, hidden=hidden, feat_dim=feat_dim).to(device)
    R = RegressorHead(feat_dim).to(device)
    D = DomainDiscriminator(feat_dim).to(device)
    CM = CurriculumManager(d).to(device)

    opt_main = optim.Adam(list(F.parameters()) + list(R.parameters()), lr=lr)
    opt_disc = optim.Adam(D.parameters(), lr=lr)
    opt_cm = optim.Adam(CM.parameters(), lr=lr)

    n_iter = max(1, num_epochs)

    for t in range(1, n_iter + 1):
        p = t / n_iter
        lam = 2.0 / (1.0 + np.exp(-gamma * p)) - 1.0

        # ----------------------------------------------------
        # 1) Update discriminator
        # ----------------------------------------------------
        with torch.no_grad():
            zsrc = F(Xs_t)
            ztgt = F(Xt_t)
            wsrc = CM(Xs_t)

        logits_s = D(zsrc)
        logits_t = D(ztgt)
        y_s = torch.ones_like(logits_s)
        y_t = torch.zeros_like(logits_t)

        loss_disc = bce_logits(logits_s, y_s, weights=wsrc.detach()) + bce_logits(logits_t, y_t)
        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # ----------------------------------------------------
        # 2) Update curriculum manager to maximize discriminator error on source
        #    Eq. (2) in spirit: minimize weighted log D(F(x_s))
        # ----------------------------------------------------
        with torch.no_grad():
            zsrc_det = F(Xs_t)
        wsrc_cm = CM(Xs_t)
        logits_s_det = D(zsrc_det).detach()
        prob_s = torch.sigmoid(logits_s_det).clamp(1e-6, 1 - 1e-6)
        loss_cm = torch.mean(wsrc_cm * torch.log(prob_s))
        opt_cm.zero_grad()
        loss_cm.backward()
        opt_cm.step()

        # ----------------------------------------------------
        # 3) Update feature extractor + regressor
        # ----------------------------------------------------
        zsrc = F(Xs_t)
        ztgt = F(Xt_t)
        pred_s = R(zsrc)
        pred_t = R(ztgt)

        reg_loss = nn.functional.mse_loss(pred_s, Ys_t) + lambda_tgt * nn.functional.mse_loss(pred_t, Yt_t)

        wsrc = CM(Xs_t).detach()
        logits_s = D(zsrc)
        logits_t = D(ztgt)
        adv_loss = bce_logits(logits_s, torch.zeros_like(logits_s), weights=wsrc) + \
                   bce_logits(logits_t, torch.ones_like(logits_t))

        loss_main = reg_loss + lambda_adv * lam * adv_loss
        opt_main.zero_grad()
        loss_main.backward()
        opt_main.step()

    F.eval()
    R.eval()
    with torch.no_grad():
        yhat = R(F(Xtest_t)).cpu().numpy().reshape(-1)
    return yhat


# ============================================================
# Other existing methods 

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


def rbf_kernel(X, Y, gamma=0.4):
    X = X if X.ndim == 2 else X.view(X.size(0), -1)
    Y = Y if Y.ndim == 2 else Y.view(Y.size(0), -1)
    XX = torch.sum(X ** 2, 1).view(-1, 1)
    YY = torch.sum(Y ** 2, 1).view(1, -1)
    distances = XX + YY - 2 * torch.mm(X, Y.t())
    K = torch.exp(-gamma * distances)
    return K


def MLcon_kernel(source_list, source_pred, target_list, target_y, lamda=1.0):
    X_p = source_list[0]
    Y_p = source_pred
    X_q = target_list[0]
    Y_q = target_y

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


def train_feature_extractor(F, X, Y, domain_ids, device, epochs=1000, lr=1e-3):
    F = F.to(device)
    optimizer = optim.Adam(F.parameters(), lr=lr)
    X = torch.tensor(X, dtype=torch.float32).to(device)
    Y = torch.tensor(Y, dtype=torch.float32).to(device)
    domain_ids = torch.tensor(domain_ids, dtype=torch.long).to(device)
    for _ in range(epochs):
        F.train()
        features = F(X, domain_ids)
        loss = psp_loss(features, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return F


def train_regressor(F, R, X, Y, domain_ids, device, epochs=1000, lr=1e-3):
    F = F.to(device)
    R = R.to(device)
    optimizer = optim.Adam(R.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    X = torch.tensor(X, dtype=torch.float32).to(device)
    Y = torch.tensor(Y, dtype=torch.float32).to(device)
    domain_ids = torch.tensor(domain_ids, dtype=torch.long).to(device)
    for _ in range(epochs):
        F.eval()
        with torch.no_grad():
            features = F(X, domain_ids)
        preds = R(features)
        loss = loss_fn(preds, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return R


# ============================================================
# Main simulation loop
# ============================================================

def run_simulation(job_id):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res_full = pd.DataFrame()

    for n_0 in [50, 100, 150, 200, 250]:
        for n_s in [100, 200, 500, 1000, 2000]:
            d = 5
            np.random.seed(job_id)

            dat_source, dat0, dat0_full, dat_test = sim_data(
                n_s=n_s, n_0=n_0, n_test=5000, sig=0.5,
                mu_s=np.ones(d), mu_t=np.zeros(d), Sigma=np.eye(d),
                beta1=1 / np.arange(1, d + 1)
            )

            X_source = [dat[:, 1:] for dat in dat_source]
            Y_source = [dat[:, 0] for dat in dat_source]
            X0 = dat0[:, 1:]
            Y0 = dat0[:, 0]
            X0_test = dat_test[:, 1:]
            Y0_test = dat_test[:, 0]

            # ------------------------------------------------
            # Existing TKRR baseline
            # ------------------------------------------------
            m = len(X_source)
            X0_T1, X0_T2, Y0_T1, Y0_T2 = train_test_split(
                X0, Y0, test_size=0.5, random_state=0
            )
            X0_T21, X0_T22, Y0_T21, Y0_T22 = train_test_split(
                X0_T2, Y0_T2, test_size=0.5, random_state=0
            )

            fb0 = fit_krr(X0_T1, Y0_T1)
            fbk_list = [fit_krr(Xk, Yk) for Xk, Yk in zip(X_source, Y_source)]

            norms = [rkhs_norm(fbk, fb0, X0_T1) for fbk in fbk_list]
            ranks = np.argsort(norms)

            candidate_models = [fb0]

            for ell in range(1, m + 1):
                selected_indices = ranks[:ell]
                X_comb = np.concatenate([X_source[i] for i in selected_indices] + [X0_T1], axis=0)
                Y_comb = np.concatenate([Y_source[i] for i in selected_indices] + [Y0_T1], axis=0)
                comb_krr = fit_krr(X_comb, Y_comb)

                Y0_pred = comb_krr.predict(X0_T1)
                Y_resi = Y0_T1 - Y0_pred
                resi_krr = fit_krr(X0_T1, Y_resi)

                class CombinedModel:
                    def __init__(self, m1, m2):
                        self.m1 = m1
                        self.m2 = m2

                    def predict(self, X):
                        return self.m1.predict(X) + self.m2.predict(X)

                candidate_models.append(CombinedModel(comb_krr, resi_krr))

            risks = [np.mean((model.predict(X0_T21) - Y0_T21) ** 2) for model in candidate_models]
            best_idx = np.argmin(risks)
            min_risk = risks[best_idx]
            best_combo = (best_idx, None, 1.0)

            for i, j in combinations(range(len(candidate_models)), 2):
                preds_i = candidate_models[i].predict(X0_T21)
                preds_j = candidate_models[j].predict(X0_T21)
                for t in np.linspace(0, 1, 101):
                    preds = t * preds_i + (1 - t) * preds_j
                    risk = np.mean((preds - Y0_T21) ** 2)
                    if risk < min_risk:
                        min_risk = risk
                        best_combo = (i, j, t)

            i, j, t = best_combo

            class AggregatedModel:
                def __init__(self, m1, m2, t):
                    self.m1 = m1
                    self.m2 = m2
                    self.t = t

                def predict(self, X):
                    if self.m2 is None:
                        return self.m1.predict(X)
                    return self.t * self.m1.predict(X) + (1 - self.t) * self.m2.predict(X)

            if j is not None:
                fba = AggregatedModel(candidate_models[i], candidate_models[j], t)
            else:
                fba = AggregatedModel(candidate_models[i], None, t)

            tkrr_mse = np.mean((fba.predict(X0_test) - Y0_test) ** 2)

            # ------------------------------------------------
            # Existing CDAR baseline
            # ------------------------------------------------
            X_source_tensor, Y_source_tensor = to_tensor(X_source[0], Y_source[0], device=device)
            X0_train_tensor, Y0_train_tensor = to_tensor(X0, Y0, device=device)
            X0_test_tensor, Y0_test_tensor = to_tensor(X0_test, Y0_test, device=device)

            model = TargetCNN(d=d).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
            criterion = nn.MSELoss()
            Lambda = 1.0
            Beta = 1.0
            num_epochs = 100

            for _ in range(num_epochs):
                model.train()
                source_list, source_pred = model(X_source_tensor)
                target_list, target_pred = model(X0_train_tensor)
                ceod_loss = MLcon_kernel(source_list, source_pred, target_list, Y0_train_tensor)
                loss = Lambda * criterion(target_pred, Y0_train_tensor) + Beta * ceod_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                _, y_pred = model(X0_test_tensor)
                y_pred = y_pred.cpu().numpy().flatten()
                cdar_mse = np.mean((Y0_test - y_pred) ** 2)

            # ------------------------------------------------
            # Existing DARC baseline
            # ------------------------------------------------
            num_source_domains = len(dat_source)
            X_source_all = np.vstack([dat[:, 1:] for dat in dat_source])
            Y_source_all = np.concatenate([dat[:, 0] for dat in dat_source])
            domain_ids_source = np.concatenate([
                np.full(len(dat_source[i]), i, dtype=int) for i in range(num_source_domains)
            ])
            X_target = dat0[:, 1:]
            Y_target = dat0[:, 0]
            domain_ids_target = np.full(len(X_target), num_source_domains, dtype=int)

            X_train = np.vstack([X_source_all, X_target])
            Y_train = np.concatenate([Y_source_all, Y_target])
            domain_ids_train = np.concatenate([domain_ids_source, domain_ids_target])

            F = FeatureExtractor(input_dim=d, domain_dim=num_source_domains + 1, feature_dim=2)
            F = train_feature_extractor(F, X_train, Y_train, domain_ids_train, device=device, epochs=1000, lr=1e-3)
            R = LinearRegressor(feature_dim=2)
            R = train_regressor(F, R, X_train, Y_train, domain_ids_train, device=device, epochs=1000, lr=1e-3)

            X_test = dat_test[:, 1:]
            Y_test = dat_test[:, 0]
            domain_ids_test = np.full(len(X_test), num_source_domains, dtype=int)
            X_test_torch = torch.tensor(X_test, dtype=torch.float32).to(device)
            domain_ids_test_torch = torch.tensor(domain_ids_test, dtype=torch.long).to(device)

            with torch.no_grad():
                features_test = F(X_test_torch, domain_ids_test_torch)
                preds = R(features_test).cpu().numpy()
            darc_mse = np.mean((preds - Y_test) ** 2)

            # ------------------------------------------------
            # New reviewer-requested methods
            # ------------------------------------------------
            yhat_vncd = fit_vncd_regression(
                X_source, Y_source, X0, Y0, X0_test,
                num_epochs=300, lr=1e-3, feat_dim=16,
                lambda_disc=0.5, lambda_tgt=1.0,
                device=device, seed=job_id
            )
            vncd_mse = np.mean((yhat_vncd - Y0_test) ** 2)

            yhat_m3sda = fit_m3sda_regression(
                X_source, Y_source, X0, Y0, X0_test,
                num_epochs=300, lr=1e-3, feat_dim=16,
                lambda_moment=0.5, discrepancy_steps=3,
                device=device, seed=job_id
            )
            m3sda_mse = np.mean((yhat_m3sda - Y0_test) ** 2)

            yhat_cmss = fit_cmss_regression(
                X_source, Y_source, X0, Y0, X0_test,
                num_epochs=300, lr=1e-3, feat_dim=16,
                lambda_adv=0.5, lambda_tgt=1.0,
                device=device, seed=job_id
            )
            cmss_mse = np.mean((yhat_cmss - Y0_test) ** 2)

            mse = np.array([
                tkrr_mse, cdar_mse, darc_mse,
                vncd_mse, m3sda_mse, cmss_mse
            ])
            res_names = ['TKRR', 'CDAR', 'DARC', 'VNCD_MDD', 'M3SDA_Reg', 'CMSS_Reg']

            res_df = pd.DataFrame({'Method': res_names, 'MSE': mse})
            res_df['source_size'] = n_s
            res_df['target_size'] = n_0
            res_full = pd.concat([res_full, res_df], axis=0)

    return res_full


if __name__ == "__main__":
    job_id = int(sys.argv[1])
    print(job_id)
    res_full = run_simulation(job_id)
    out_path = f'./Results/Simulation_Concept_Covariate_{job_id}_Compare_Extended.csv'
    res_full.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
