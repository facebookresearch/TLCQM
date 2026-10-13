# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
@author: Yikun Zhang
Last Editing: Aug 1, 2026

Description: This script contains the utility functions for simulating data
under a non-spherical two-component Gaussian mixture design.
"""

import numpy as np

#=======================================================================================#


def sample_mixture_normal(n, mu, Sigma_AR, Sigma_CS, pi):
    """Generate covariates from pi*N(mu, Sigma_AR)+(1-pi)*N(-mu, Sigma_CS)."""
    component = np.random.binomial(1, pi, size=n)
    X = np.zeros((n, len(mu)))

    n_AR = np.sum(component == 1)
    n_CS = n - n_AR
    X[component == 1] = np.random.multivariate_normal(
        mean=mu, cov=Sigma_AR, size=n_AR
    )
    X[component == 0] = np.random.multivariate_normal(
        mean=-mu, cov=Sigma_CS, size=n_CS
    )
    return X


def sim_data(n_s=1000, n_0=50, n_test=5000, sig=0.5, mu=0.75*np.ones(5),
             rho_ar=0.6, rho_cs=0.3, pi_s=0.7, pi_t=0.3,
             beta1=1/np.arange(1, 6)):
    """
    Simulate source and target datasets under covariate shift and concept shift.
    The covariates follow a mixture of an AR(1) normal component and a compound
    symmetric normal component.

    Parameters
    ----------
        n_s : int
            Number of samples per source.
        n_0 : int
            Number of labeled target samples.
        n_test : int
            Number of test samples.
        sig : float
            Standard deviation of noise.
        mu : np.ndarray
            Mean vector of the first mixture component. The second component
            has mean -mu.
        rho_ar : float
            Correlation parameter of the AR(1) covariance matrix.
        rho_cs : float
            Correlation parameter of the compound symmetric covariance matrix.
        pi_s : float
            Mixture probability for the first component in the source domains.
        pi_t : float
            Mixture probability for the first component in the target domain.
        beta1 : np.ndarray
            Coefficient vector for generating responses.

    Returns
    -------
        dat_source : list of np.ndarray
            List of source datasets, each of shape (n_s, d+1).
        dat0 : np.ndarray
            Labeled target dataset of shape (n_0, d+1).
        dat0_full : np.ndarray
            Full target dataset of shape (2*n_s + n_0, d+1).
        dat_test0 : np.ndarray
            Test dataset of shape (n_test, d+1).
    """
    d = len(mu)
    index = np.arange(d)
    Sigma_AR = rho_ar ** np.abs(index[:, None] - index[None, :])
    Sigma_CS = (1-rho_cs)*np.eye(d) + rho_cs*np.ones((d, d))

    # Target data
    X_dat0 = sample_mixture_normal(n_0, mu, Sigma_AR, Sigma_CS, pi_t)
    Y0 = np.sin(3*np.dot(X_dat0, beta1))/3 - 1 + np.random.randn(n_0)*sig
    dat0 = np.column_stack([Y0, X_dat0])

    # Source data
    X_dat1 = sample_mixture_normal(n_s, mu, Sigma_AR, Sigma_CS, pi_s)
    Y1 = np.sin(3*np.dot(X_dat1, beta1)) + 1 + np.random.randn(n_s)*sig
    dat1 = np.column_stack([Y1, X_dat1])

    X_dat2 = sample_mixture_normal(n_s, mu, Sigma_AR, Sigma_CS, pi_s)
    Y2 = 2*np.cos(3*np.dot(X_dat2, beta1)) + 1 + np.random.randn(n_s)*sig
    dat2 = np.column_stack([Y2, X_dat2])

    dat_source = [dat1, dat2]

    X_dat0_full = sample_mixture_normal(2*n_s+n_0, mu, Sigma_AR, Sigma_CS, pi_t)
    Y0_full = np.sin(3*np.dot(X_dat0_full, beta1))/3 - 1 + np.random.randn(2*n_s+n_0)*sig
    dat0_full = np.column_stack([Y0_full, X_dat0_full])

    X_test0 = sample_mixture_normal(n_test, mu, Sigma_AR, Sigma_CS, pi_t)
    Y0_test = np.sin(3*np.dot(X_test0, beta1))/3 - 1 + np.random.randn(n_test)*sig
    dat_test0 = np.column_stack([Y0_test, X_test0])

    return dat_source, dat0, dat0_full, dat_test0
