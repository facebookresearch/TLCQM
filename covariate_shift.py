# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
@author: Yikun Zhang
Last Editing: Jan 15, 2025

Description: Implementation of Kernel Mean Matching for covariate shift adjustment
adopted from https://github.com/vodp/py-kmm.
References:
    1. Gretton, Arthur, et al. "Covariate shift by kernel mean matching." Dataset shift in machine learning 3.4 (2009): 5.
    2. Huang, Jiayuan, et al. "Correcting sample selection bias by unlabeled data." Advances in neural information processing systems. 2006.
"""

# =======================================================================================#

import numpy as np
from cvxopt import matrix, solvers


def kernel_mean_matching(X, Z, kern="lin", B=1.0, eps=None):
    """
    Kernel Mean Matching for covariate shift adjustment.

    Parameters
    ----------
        X : np.ndarray
            Source samples of shape (nx, d).
        Z : np.ndarray
            Target samples of shape (nz, d).
        kern : str
            Kernel type, either "lin" for linear or "rbf" for radial basis function.
        B : float
            Upper bound on the weights.
        eps : float or None
            Tolerance parameter. If None, it is set to B / sqrt(nz).

    Returns
    -------
        coef : np.ndarray
            Importance weights for source samples.
    """
    nx = X.shape[0]
    nz = Z.shape[0]
    if eps is None:
        eps = B / np.sqrt(nz)
    if kern == "lin":
        K = np.dot(Z, Z.T)
        kappa = np.sum(np.dot(Z, X.T) * float(nz) / float(nx), axis=1)
    elif kern == "rbf":
        K = compute_rbf(Z, Z)
        kappa = np.sum(compute_rbf(Z, X), axis=1) * float(nz) / float(nx)
    else:
        raise ValueError("unknown kernel")

    K = matrix(K)
    kappa = matrix(kappa)
    G = matrix(np.r_[np.ones((1, nz)), -np.ones((1, nz)), np.eye(nz), -np.eye(nz)])
    h = matrix(
        np.r_[nz * (1 + eps), nz * (eps - 1), B * np.ones((nz,)), np.zeros((nz,))]
    )

    solvers.options["show_progress"] = False
    sol = solvers.qp(K, -kappa, G, h)
    coef = np.array(sol["x"])
    return coef


def compute_rbf(X, Z, sigma=1.0):
    K = np.zeros((X.shape[0], Z.shape[0]), dtype=float)
    for i, vx in enumerate(X):
        K[i, :] = np.exp(-np.sum((vx - Z) ** 2, axis=1) / (2.0 * sigma))
    return K
