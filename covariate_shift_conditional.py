# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
@author: Yikun Zhang
Last Editing: Jul 25, 2026

Description: Robust kernel mean matching for the conditional TLCQM experiments.
"""

import warnings

import numpy as np

try:
    from cvxopt import matrix, solvers
except ImportError:
    matrix = solvers = None


def compute_rbf(X, Z, sigma=1.0):
    X_norm = np.sum(X ** 2, axis=1)[:, None]
    Z_norm = np.sum(Z ** 2, axis=1)[None, :]
    distance = np.maximum(X_norm + Z_norm - 2 * X @ Z.T, 0)
    return np.exp(-distance / (2 * sigma))


def _standardize_covariates(X, Z):
    pooled = np.concatenate([X, Z], axis=0)
    center = np.mean(pooled, axis=0)
    scale = np.std(pooled, axis=0)
    scale[scale < 1e-12] = 1.0
    return (X - center) / scale, (Z - center) / scale


def _median_distance(X, Z, max_samples=1000):
    def subsample(A, n):
        if A.shape[0] <= n:
            return A
        idx = np.linspace(0, A.shape[0] - 1, n, dtype=int)
        return A[idx]

    n = max_samples // 2
    pooled = np.concatenate([subsample(X, n), subsample(Z, n)], axis=0)
    norm = np.sum(pooled ** 2, axis=1)
    distance = np.maximum(norm[:, None] + norm[None, :] - 2 * pooled @ pooled.T, 0)
    distance = distance[np.triu_indices(pooled.shape[0], k=1)]
    distance = distance[distance > 1e-12]
    return float(np.median(distance)) if distance.size else 1.0


def _is_feasible(beta, nz, B, eps, tol=1e-5):
    if beta is None or beta.size != nz or not np.all(np.isfinite(beta)):
        return False
    lower, upper = nz * (1 - eps), nz * (1 + eps)
    sum_tol = tol * max(1, nz)
    return (
        np.min(beta) >= -tol
        and np.max(beta) <= B + tol
        and lower - sum_tol <= np.sum(beta) <= upper + sum_tol
    )


def _solver_diagnostics(result):
    def value(key):
        x = result.get(key)
        return np.inf if x is None else float(x)

    relative_gap = result.get("relative gap")
    relative_gap = np.inf if relative_gap is None else float(relative_gap)
    return {
        "primal": value("primal infeasibility"),
        "dual": value("dual infeasibility"),
        "gap": value("gap"),
        "relative_gap": relative_gap,
    }


def _acceptable_unknown(result, beta, nz, B, eps):
    diag = _solver_diagnostics(result)
    objective = result.get("primal objective")
    objective = 0.0 if objective is None else abs(float(objective))
    gap_ok = (
        diag["relative_gap"] <= 1e-4
        or diag["gap"] <= 1e-6 * max(1.0, objective)
    )
    return (
        _is_feasible(beta, nz, B, eps)
        and diag["primal"] <= 1e-6
        and diag["dual"] <= 1e-4
        and gap_ok
    )


def _project_weights(beta, nz, B, eps):
    lower = max(0.0, nz * (1 - eps))
    upper = min(B * nz, nz * (1 + eps))
    projected = np.clip(beta, 0, B)
    if lower <= np.sum(projected) <= upper:
        return projected

    target = lower if np.sum(projected) < lower else upper
    left, right = np.min(beta - B), np.max(beta)
    for _ in range(60):
        shift = (left + right) / 2
        projected = np.clip(beta - shift, 0, B)
        if np.sum(projected) > target:
            left = shift
        else:
            right = shift
    return np.clip(beta - (left + right) / 2, 0, B)


def _solve_projected_gradient(K, kappa, nz, B, eps):
    beta = _project_weights(np.ones(nz), nz, B, eps)
    extrapolated = beta.copy()
    momentum = 1.0
    lipschitz = 1.05 * np.max(np.sum(np.abs(K), axis=1))
    lipschitz = max(lipschitz, 1e-8)

    for _ in range(1000):
        candidate = _project_weights(
            extrapolated - (K @ extrapolated - kappa) / lipschitz,
            nz,
            B,
            eps,
        )
        if np.max(np.abs(candidate - beta)) <= 1e-7 * (1 + np.max(np.abs(beta))):
            beta = candidate
            break
        next_momentum = (1 + np.sqrt(1 + 4 * momentum ** 2)) / 2
        extrapolated = candidate + (momentum - 1) / next_momentum * (
            candidate - beta
        )
        beta, momentum = candidate, next_momentum

    residual = np.max(
        np.abs(
            beta
            - _project_weights(
                beta - (K @ beta - kappa) / lipschitz, nz, B, eps
            )
        )
    )
    if residual > 1e-4:
        warnings.warn(
            f"Projected-gradient KMM stopped with residual {residual:.2e}.",
            RuntimeWarning,
        )
    if not _is_feasible(beta, nz, B, eps):
        raise RuntimeError("Projected-gradient KMM returned infeasible weights.")
    return beta.reshape(-1, 1)


def kernel_mean_matching(X, Z, kern="lin", B=1.0, eps=None, sigma=None):
    """
    Estimate weights for rows of Z so their weighted mean embedding matches X.
    The two samples are standardized together before constructing the kernel.
    """
    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=float)
    if X.ndim != 2 or Z.ndim != 2 or X.shape[1] != Z.shape[1]:
        raise ValueError("X and Z must be two-dimensional with the same columns.")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Z)):
        raise ValueError("X and Z must contain only finite values.")

    X, Z = _standardize_covariates(X, Z)
    nx, nz = X.shape[0], Z.shape[0]
    if eps is None:
        eps = B / np.sqrt(nz)
    if B <= 0 or eps < 0 or B * nz < nz * (1 - eps):
        raise ValueError(
            "The KMM constraints are infeasible for the supplied B and eps."
        )

    if kern == "lin":
        K = Z @ Z.T
        kappa = np.sum(Z @ X.T, axis=1) * nz / nx
    elif kern == "rbf":
        if sigma is None:
            sigma = _median_distance(X, Z)
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        K = compute_rbf(Z, Z, sigma)
        kappa = np.sum(compute_rbf(Z, X, sigma), axis=1) * nz / nx
    else:
        raise ValueError("kern must be 'lin' or 'rbf'.")

    K = (K + K.T) / 2
    if matrix is None:
        K.flat[::nz + 1] += 1e-4
        return _solve_projected_gradient(K, kappa, nz, B, eps)

    G = matrix(
        np.r_[np.ones((1, nz)), -np.ones((1, nz)), np.eye(nz), -np.eye(nz)]
    )
    h = matrix(
        np.r_[nz * (1 + eps), nz * (eps - 1), B * np.ones(nz), np.zeros(nz)]
    )
    old_options = dict(solvers.options)
    last_result = None
    try:
        solvers.options["show_progress"] = False
        solvers.options["maxiters"] = 200
        for ridge in [1e-6, 1e-4]:
            K_ridge = K.copy()
            K_ridge.flat[::nz + 1] += ridge
            result = solvers.qp(matrix(K_ridge), matrix(-kappa), G, h)
            beta = np.asarray(result["x"]).reshape(-1)
            if result["status"] == "optimal" and _is_feasible(beta, nz, B, eps):
                return np.clip(beta, 0, B).reshape(-1, 1)
            if result["status"] == "unknown" and _acceptable_unknown(
                result, beta, nz, B, eps
            ):
                warnings.warn(
                    "CVXOPT returned 'unknown', but its final KMM iterate satisfies "
                    "the feasibility, residual, and duality-gap tolerances.",
                    RuntimeWarning,
                )
                return np.clip(beta, 0, B).reshape(-1, 1)
            last_result = result
    except Exception as error:
        warnings.warn(
            "CVXOPT failed during KMM; retrying with projected gradient. "
            + str(error),
            RuntimeWarning,
        )
    finally:
        solvers.options.clear()
        solvers.options.update(old_options)

    if last_result is not None:
        diag = _solver_diagnostics(last_result)
        warnings.warn(
            "CVXOPT did not meet the KMM accuracy tolerances "
            f"(status={last_result['status']}, primal={diag['primal']:.2e}, "
            f"dual={diag['dual']:.2e}, relative_gap={diag['relative_gap']:.2e}); "
            "retrying with projected gradient.",
            RuntimeWarning,
        )
    K.flat[::nz + 1] += 1e-4
    return _solve_projected_gradient(K, kappa, nz, B, eps)
