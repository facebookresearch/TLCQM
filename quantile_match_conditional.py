# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
@author: Yikun Zhang
Last Editing: Jul 25, 2026

Description: Conditional CRPS matching for the revised TLCQM estimator.
"""

import numpy as np
from scipy.optimize import OptimizeResult, minimize


def _check_input(target, source_sample):
    target = np.asarray(target, dtype=float).reshape(-1)
    source_sample = np.asarray(source_sample, dtype=float)
    if source_sample.ndim != 3:
        raise ValueError("source_sample must have shape (n_0, M, K).")
    if source_sample.shape[0] != len(target):
        raise ValueError("The first dimension must match the target sample.")
    if source_sample.shape[1] < 2:
        raise ValueError("At least two draws per target covariate are required.")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(source_sample)):
        raise ValueError("The inputs must contain only finite values.")
    return target, source_sample


def _add_intercept(source_sample):
    n_0, M = source_sample.shape[:2]
    return np.concatenate([np.ones((n_0, M, 1)), source_sample], axis=2)


def _project_l1(beta, radius, positive=False):
    beta = np.maximum(beta, 0) if positive else np.asarray(beta, dtype=float)
    if np.sum(np.abs(beta)) <= radius:
        return beta
    u = np.sort(np.abs(beta))[::-1]
    rho = np.where(u * np.arange(1, len(u) + 1) >
                   np.cumsum(u) - radius)[0][-1]
    theta = (np.sum(u[:rho + 1]) - radius) / (rho + 1)
    return np.sign(beta) * np.maximum(np.abs(beta) - theta, 0)


def _first_term(beta, target, design):
    z = np.einsum("nmp,p->nm", design, beta)
    loss = np.mean(np.abs(z - target[:, None]))
    grad = np.mean(
        np.sign(z - target[:, None])[:, :, None] * design, axis=1
    )
    return float(loss), np.mean(grad, axis=0)


def _second_term(beta, design):
    n_0, M = design.shape[:2]
    z = np.einsum("nmp,p->nm", design, beta)
    order = np.argsort(z, axis=1)
    z_sort = np.take_along_axis(z, order, axis=1)
    v_sort = np.take_along_axis(design, order[:, :, None], axis=1)
    rank = 2 * np.arange(M) - M + 1

    loss = np.mean(np.sum(z_sort * rank, axis=1) / (M * (M - 1)))
    grad = np.mean(
        np.sum(v_sort * rank[None, :, None], axis=1) / (M * (M - 1)),
        axis=0,
    )
    return float(loss), grad


def conditional_crps_loss_and_grad(beta, target, design):
    """
    Conditional CRPS and a DC subgradient, using within-covariate sorting.
    """
    first, first_grad = _first_term(beta, target, design)
    second, second_grad = _second_term(beta, design)
    return first - second, first_grad - second_grad


def _run_proximal_dca(
    beta,
    target,
    design,
    constraint,
    bounds,
    stop_eps,
    max_iter,
    proximal,
    verbose,
):
    current = beta.copy()
    current_loss = conditional_crps_loss_and_grad(
        current, target, design
    )[0]
    history = [current_loss]
    message = "Maximum number of iterations reached."

    for iteration in range(int(max_iter)):
        anchor = current.copy()
        _, second_grad = _second_term(anchor, design)

        def surrogate(beta_new):
            first, first_grad = _first_term(beta_new, target, design)
            difference = beta_new - anchor
            loss = (
                first - second_grad @ beta_new
                + 0.5 * proximal * difference @ difference
            )
            grad = first_grad - second_grad + proximal * difference
            return loss, grad

        anchor_surrogate = surrogate(anchor)[0]
        result = minimize(
            surrogate,
            anchor,
            method="SLSQP",
            jac=True,
            bounds=bounds,
            constraints=constraint,
            options={
                "maxiter": 1000,
                "ftol": max(float(stop_eps) * 0.1, 1e-12),
                "disp": False,
            },
        )
        candidate = np.asarray(result.x, dtype=float)
        candidate_surrogate = surrogate(candidate)[0]

        if not np.isfinite(candidate_surrogate):
            message = "The convex surrogate solver returned a nonfinite value."
            break

        if candidate_surrogate > anchor_surrogate:
            direction = candidate - anchor
            for step in 0.5 ** np.arange(1, 31):
                trial = anchor + step * direction
                if surrogate(trial)[0] <= anchor_surrogate:
                    candidate = trial
                    break
            else:
                message = "No decreasing surrogate step was found."
                break

        new_loss = conditional_crps_loss_and_grad(
            candidate, target, design
        )[0]
        if new_loss > current_loss + 1e-10 * (1 + abs(current_loss)):
            message = "The accepted surrogate step increased the CRPS."
            break
        difference = np.linalg.norm(candidate - anchor)
        decrease = current_loss - new_loss
        current, current_loss = candidate, new_loss
        history.append(current_loss)

        if verbose:
            print(
                "DCA iteration {}, loss: {:.8f}".format(
                    iteration + 1, current_loss
                )
            )
        if (
            difference <= stop_eps * (1 + np.linalg.norm(anchor))
            or decrease <= stop_eps * (1 + abs(current_loss))
        ):
            message = "Proximal DCA converged."
            break

    return OptimizeResult(
        x=current,
        fun=current_loss,
        nit=len(history) - 1,
        success=np.isfinite(current_loss),
        message=message,
        history=np.asarray(history),
    )


def conditional_crps_score(target, source_sample, beta):
    """Evaluate the empirical conditional CRPS at beta."""
    target, source_sample = _check_input(target, source_sample)
    design = _add_intercept(source_sample)
    beta = np.asarray(beta, dtype=float).reshape(-1)
    if len(beta) != design.shape[2]:
        raise ValueError("beta has the wrong length.")
    return conditional_crps_loss_and_grad(beta, target, design)[0]


def conditional_crps_estimate(
    target,
    source_sample,
    beta_init=None,
    stop_eps=1e-8,
    max_iter=1000,
    positive=False,
    beta_bound=10.0,
    n_restarts=10,
    proximal=1e-4,
    random_state=None,
    verbose=False,
    return_result=False,
):
    """
    Estimate beta over {beta: ||beta||_1 <= beta_bound} by proximal DCA.

    source_sample has shape (n_0, M, K), so target response i is compared
    only with draws generated at target covariate i. If positive=True, all
    coefficients, including the intercept, are constrained to be nonnegative.
    The restarts comprise one least-squares start and random feasible starts.
    """
    target, source_sample = _check_input(target, source_sample)
    design = _add_intercept(source_sample)
    p = design.shape[2]

    if beta_bound is None or beta_bound <= 0:
        raise ValueError("beta_bound must be a positive finite number.")
    if n_restarts < 1:
        raise ValueError("n_restarts must be at least one.")
    if proximal <= 0:
        raise ValueError("proximal must be positive.")

    if beta_init is None:
        source_mean = np.mean(source_sample, axis=1)
        mean_design = np.column_stack([np.ones(len(target)), source_mean])
        beta_init = np.linalg.lstsq(mean_design, target, rcond=None)[0]
    beta_init = np.asarray(beta_init, dtype=float).reshape(-1)
    if len(beta_init) != p:
        raise ValueError("beta_init must contain K+1 coefficients.")
    beta_init = _project_l1(beta_init, beta_bound, positive=positive)

    rng = np.random.default_rng(random_state)
    beta_start = [beta_init]
    for _ in range(n_restarts - 1):
        beta = rng.normal(size=p)
        if positive:
            beta = np.abs(beta)
        beta *= rng.uniform(0.25, 1.0) * beta_bound / np.sum(np.abs(beta))
        beta_start.append(beta)

    bounds = [(0, None)] * p if positive else None
    constraint = {
        "type": "ineq",
        "fun": lambda beta: beta_bound - np.sum(np.abs(beta)),
        "jac": lambda beta: -np.sign(beta),
    }

    result_all = []
    for beta in beta_start:
        result = _run_proximal_dca(
            beta,
            target,
            design,
            constraint,
            bounds,
            stop_eps,
            max_iter,
            proximal,
            verbose,
        )
        result_all.append(result)

    result_all = [
        result for result in result_all
        if result.success and np.isfinite(result.fun)
    ]
    if not result_all:
        raise RuntimeError("All conditional CRPS optimization runs failed.")
    result = min(result_all, key=lambda x: x.fun)
    beta_hat = _project_l1(
        np.asarray(result.x, dtype=float), beta_bound, positive=positive
    )
    result.x = beta_hat
    result.fun = conditional_crps_loss_and_grad(
        beta_hat, target, design
    )[0]
    if return_result:
        return beta_hat, result
    return beta_hat
