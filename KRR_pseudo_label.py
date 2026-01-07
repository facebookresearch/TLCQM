#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code adopted from https://github.com/kw2934/KRR for comparison purposes.
"""

import numpy as np
from sklearn.kernel_ridge import KernelRidge


def Kernel(X, Y):  # compute kernel matrix given two sets of covariates
    """
    Compute the min kernel matrix between two sets of vectors.

    Args:
        X: np.ndarray of shape (n_1, d)
        Y: np.ndarray of shape (n_2, d)

    Returns:
        K: np.ndarray of shape (n_1, n_2)
    """
    n_1, d = X.shape
    n_2 = Y.shape[0]
    K = np.zeros((n_1, n_2))
    for i in range(n_1):
        for j in range(n_2):
            K[i, j] = np.mean(np.minimum(X[i], Y[j]))
    return K


def f_star(x, fcn):  # get true regression function
    if fcn == "C":
        return np.cos(x * 2 * np.pi) - 1
    if fcn == "S":
        return np.sin(x * 2 * np.pi)
    if fcn == "V":
        return np.abs(x - 1 / 2) - 1 / 2
    if fcn == "W":
        f_1 = np.clip(np.abs(4 * x - 1), 0, 1)
        f_2 = np.clip(np.abs(4 * x - 3), 0, 1)
        return f_1 + f_2 - 2
    if fcn == "x":
        return x * np.sin(4 * np.pi * x)


class KRR_covariate_shift:
    def __init__(
        self, n, n_0, B, sigma, X, y, X_0, y_0, seed
    ):  # preparations, sample generation
        assert B >= 1
        np.random.seed(seed)
        self.B = B

        # source data
        self.n = n
        self.X = X
        self.y = y

        # target data
        self.X_0 = X_0
        self.y_0 = y_0

    def fit(self, rho=0.5, beta=2):  # KRR under covariate shift
        assert beta > 1
        assert 0 < rho and rho < 1

        # data splitting
        self.n_1 = int((1 - rho) * self.n)
        self.n_2 = self.n - self.n_1
        self.X_1, self.y_1 = self.X[0 : self.n_1], self.y[0 : self.n_1]
        self.X_2, self.y_2 = self.X[self.n_1 : self.n], self.y[self.n_1 : self.n]

        # penalty parameters: one for imputation, a geometric sequence for training
        lbd_tilde = 0.1 / self.n  # for the imputation model
        lbd_min, lbd_max = 0.1 / self.n, 1  # min and max for training
        m = np.log(lbd_max / lbd_min) / np.log(beta)
        m = max(int(np.ceil(m)), 2) + 1
        self.Lambda = lbd_min * (beta ** np.array(range(m)))  # for training

        # pseudo-labeling (call the KRR solver in sklearn)
        krr = KernelRidge(kernel="precomputed", alpha=lbd_tilde)
        krr.fit(Kernel(self.X_2, self.X_2), self.y_2)
        self.alpha_tilde = krr.dual_coef_
        a = Kernel(self.X_0, self.X_2) @ self.alpha_tilde
        self.y_tilde = a * (a > 0)

        # training (call the KRR solver in sklearn)
        self.Alpha = np.zeros((m, self.n_1))
        self.err_est_naive = np.zeros(m)
        self.err_est_pseudo = np.zeros(m)
        self.err_est_real = np.zeros(m)
        for j, lbd in enumerate(self.Lambda):
            # KRR
            krr = KernelRidge(kernel="precomputed", alpha=lbd)
            krr.fit(Kernel(self.X_1, self.X_1), self.y_1)
            self.Alpha[j] = krr.dual_coef_

            # naive estimate of loss (using source data)
            self.err_est_naive[j] = np.mean(
                (Kernel(self.X_2, self.X_1) @ self.Alpha[j] - self.y_2) ** 2
            )

            # pseudo and real labels
            y_lbd = Kernel(self.X_0, self.X_1) @ self.Alpha[j]
            self.err_est_pseudo[j] = np.mean((y_lbd - self.y_tilde) ** 2)
            self.err_est_real[j] = np.mean((y_lbd - self.y_0) ** 2)

        # selection
        self.j_naive = np.argmin(self.err_est_naive)
        self.lbd_naive = self.Lambda[self.j_naive]
        self.alpha_naive = self.Alpha[self.j_naive]

        self.j_pseudo = np.argmin(self.err_est_pseudo)
        self.lbd_pseudo = self.Lambda[self.j_pseudo]
        self.alpha_pseudo = self.Alpha[self.j_pseudo]

        self.j_real = np.argmin(self.err_est_real)
        self.lbd_real = self.Lambda[self.j_real]
        self.alpha_real = self.Alpha[self.j_real]

    ############################################
    # evaluation of candidates
    def predict_candidates(
        self, X_new, list_idx_candidates
    ):  # make predictions using candidates
        self.y_new_true = f_star(X_new, self.fcn)
        K = Kernel(X_new, self.X_1)
        m = len(list_idx_candidates)
        self.y_new_candidates = []
        for j in list_idx_candidates:
            self.y_new_candidates.append(K @ self.Alpha[j])

    def evaluate_candidates(
        self, distribution, list_idx_candidates, N_test, seed
    ):  # evaluate excess risk on the source or the target distribution
        np.random.seed(seed)
        tmp = int(N_test * self.B / (self.B + 1))
        if distribution == "target":
            self.X_test_0 = np.concatenate(
                (np.random.rand(N_test - tmp) / 2, 1 / 2 + np.random.rand(tmp) / 2)
            )
        elif distribution == "source":
            self.X_test_0 = np.concatenate(
                (np.random.rand(tmp) / 2, 1 / 2 + np.random.rand(N_test - tmp) / 2)
            )

        self.predict_candidates(self.X_test_0, list_idx_candidates)
        self.err_candidates = []
        self.err_candidates_ste = []
        sqrt_N = np.sqrt(N_test)
        for i in range(len(list_idx_candidates)):
            tmp = (self.y_new_true - self.y_new_candidates[i]) ** 2
            self.err_candidates.append(np.mean(tmp))
            self.err_candidates_ste.append(np.std(tmp) / sqrt_N)

    ############################################
    # evaluation of selected models
    def predict_final(self, X_new):  # make predictions using selected models
        a = Kernel(X_new, self.X_2) @ self.alpha_tilde
        self.y_new_tilde = a * (a > 0)

        K = Kernel(X_new, self.X_1)
        self.y_new_naive = K @ self.alpha_naive
        self.y_new_pseudo = K @ self.alpha_pseudo
        self.y_new_real = K @ self.alpha_real

    def evaluate_final(
        self, N_test, seed
    ):  # evaluate excess risk on the target distribution using selected models and newly generated samples
        np.random.seed(seed)
        tmp = int(N_test * self.B / (self.B + 1))
        self.X_test_0 = np.concatenate(
            (np.random.rand(N_test - tmp) / 2, 1 / 2 + np.random.rand(tmp) / 2)
        )
        self.predict_final(self.X_test_0)

        sqrt_N = np.sqrt(N_test)

        tmp = (self.y_new_true - self.y_new_naive) ** 2
        self.err_naive = np.mean(tmp)
        self.err_naive_ste = np.std(tmp) / sqrt_N

        tmp = (self.y_new_true - self.y_new_pseudo) ** 2
        self.err_pseudo = np.mean(tmp)
        self.err_pseudo_ste = np.std(tmp) / sqrt_N

        tmp = (self.y_new_true - self.y_new_real) ** 2
        self.err_real = np.mean(tmp)
        self.err_real_ste = np.std(tmp) / sqrt_N
