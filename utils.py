# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
@author: Yikun Zhang
Last Editing: Jan 15, 2026

Description: This script contains the utility functions for simulating data.
"""

import numpy as np

#=======================================================================================#


def sim_data(n_s=1000, n_0=50, n_test=5000, sig=0.5, mu_s=np.ones(5), mu_t=np.zeros(5), 
             Sigma=np.eye(5), beta1=1/np.arange(1, 6)):
    """
    Simulate source and target datasets under covariate shift and concept shift.

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
        mu_s : np.ndarray
            Mean vector for source covariates.
        mu_t : np.ndarray
            Mean vector for target covariates.
        Sigma : np.ndarray
            Covariance matrix for covariates.
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
    # Target data
    X_dat0 = np.random.multivariate_normal(mean=mu_t, cov=0.25*Sigma, size=n_0)
    Y0 = np.sin(3*np.dot(X_dat0, beta1))/3 - 1 + np.random.randn(n_0)*sig
    dat0 = np.column_stack([Y0, X_dat0])
    
    # Source data
    X_dat1 = np.random.multivariate_normal(mean=mu_s, cov=Sigma, size=n_s)
    Y1 = np.sin(3*np.dot(X_dat1, beta1)) + 1 + np.random.randn(n_s)*sig
    dat1 = np.column_stack([Y1, X_dat1])

    X_dat2 = np.random.multivariate_normal(mean=mu_s, cov=Sigma, size=n_s)
    Y2 = 2*np.cos(3*np.dot(X_dat2, beta1)) + 1 + np.random.randn(n_s)*sig
    dat2 = np.column_stack([Y2, X_dat2])

    dat_source = [dat1, dat2]

    X_dat0_full = np.random.multivariate_normal(mean=mu_t, cov=0.25*Sigma, size=2*n_s+n_0)
    Y0_full = np.sin(3*np.dot(X_dat0_full, beta1))/3 - 1 + np.random.randn(2*n_s+n_0)*sig
    dat0_full = np.column_stack([Y0_full, X_dat0_full])

    X_test0 = np.random.multivariate_normal(mean=mu_t, cov=0.25*Sigma, size=n_test)
    Y0_test = np.sin(3*np.dot(X_test0, beta1))/3 - 1 + np.random.randn(n_test)*sig
    dat_test0 = np.column_stack([Y0_test, X_test0])

    return dat_source, dat0, dat0_full, dat_test0