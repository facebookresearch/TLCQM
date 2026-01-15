# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
@author: Yikun Zhang
Last Editing: Jan 15, 2025

Description: Code for implementing quantile matching estimator.
"""

import numpy as np
from sklearn.linear_model import LinearRegression


def quantile_matching_estimate(
    target,
    source_mat,
    beta_init=None,
    stop_eps=1e-8,
    max_iter=1000,
    positive=False,
    verbose=False,
):
    if beta_init is None:
        lr_mod = LinearRegression(fit_intercept=False, positive=positive).fit(
            source_mat, target
        )
        cur_beta = lr_mod.coef_
    else:
        cur_beta = beta_init
    target_sorted = np.sort(target)
    new_loss = 1e10
    for i in range(max_iter):
        if verbose:
            print("Current beta: {}".format(cur_beta))
        cur_loss = new_loss
        source_order = np.argsort(np.dot(source_mat, cur_beta))
        order_mat = source_mat[source_order, :]
        lr_quantile = LinearRegression(fit_intercept=False, positive=positive).fit(
            order_mat, target_sorted
        )
        cur_beta = lr_quantile.coef_
        new_loss = np.mean((target_sorted - np.dot(order_mat, cur_beta)) ** 2)
        if abs(cur_loss - new_loss) < stop_eps:
            break
        if verbose:
            print("Iteration {}, loss: {:.4f}".format(i, new_loss))
    if i == max_iter - 1:
        print("Warning: maximum iteration reached.")
    return cur_beta


def direct_quantile_matching(target, source):
    sorted_target = np.sort(target)
    sorted_source = np.sort(source)
    from scipy.interpolate import interp1d

    quantiles = np.linspace(0, 1, len(target))
    interp_target = interp1d(
        quantiles, sorted_target, bounds_error=False, fill_value="extrapolate"
    )

    quantiles_source = np.searchsorted(sorted_source, source) / len(sorted_source)
    return interp_target(quantiles_source)
