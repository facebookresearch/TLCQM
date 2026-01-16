# Transfer Learning Through Conditional Quantile Matching (TLCQM)

This repository contains the Python implementation for high-quality data augmentation and synthetic data generation in transfer learning tasks.

**Paper Reference**: Y. Zhang, S. Wilkins-Reeves, W. Lee, and A. Hofleitner. *Transfer Learning Through Conditional Quantile Matching.* (2026+).

## Overview

TLCQM is a framework that addresses both **covariate shift** and **concept shift** between source and target domains. It leverages conditional quantile matching to calibrate generated samples from multiple source domains to match the target domain distribution, enabling effective transfer learning with limited labeled target data.

## Requirements

- Python >= 3.10 (earlier versions might be applicable)
- [NumPy](http://www.numpy.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [PyTorch](https://pytorch.org/) (for neural network models and auto-differentiation)
- [engression](https://github.com/xwshen51/engression/tree/main/engression-python)
- [cvxopt](https://github.com/cvxopt/cvxopt)
- Optional: [pandas](https://pandas.pydata.org/) and [Matplotlib](https://matplotlib.org/) (for data processing and plotting)

## File Descriptions

### Core Modules

| File | Description |
|------|-------------|
| `TLCQM.py` | Main implementation of the TLCQM framework with the `fit_TLCQM()` function |
| `quantile_match.py` | Quantile matching estimator via an iterative procedure |
| `covariate_shift.py` | Utilities for handling covariate shift between domains |
| `utils.py` | Utility functions for data simulation |

### Simulation Scripts

| File | Description |
|------|-------------|
| `Sim_TLCQM.py` | Simulation code I for TLCQM |
| `Sim_TLCQM_Ratio.py` | Simulation code II for TLCQM |
| `Sim_Compare.py` | Comparison code I with baseline methods |
| `Sim_Compare_Ratio.py` | Comparison code I with baseline methods |
| `Syn_Sim_Res.py` | Synthetic simulation result processing |

### Real-World Data Experiments

| File | Description |
|------|-------------|
| `Apartment_TLCQM.py` | TLCQM experiments on apartment rental data |
| `Apartment_Compare.py` | Baseline comparisons on apartment data |

### Visualization

| File | Description |
|------|-------------|
| `Figure_Plotting.ipynb` | Jupyter notebook for generating publication figures |

### SLURM Batch Scripts

Each `.py` experiment file has a corresponding `.sbatch` file for HPC cluster submission.

## Usage

### Basic Example

```python
import numpy as np
import torch
from TLCQM import fit_TLCQM
from utils import sim_data

# Generate synthetic data with covariate and concept shift
dat_source, dat_target, dat0_full, dat_test0 = sim_data(
    n_s=1000,           # samples per source
    n_0=50,             # labeled target samples
    n_test=5000,        # test samples
    sig=0.5,            # noise std
    mu_s=np.ones(5),    # source covariate mean
    mu_t=np.zeros(5),   # target covariate mean
    Sigma=np.eye(5),    # covariate covariance
    beta1=1/np.arange(1, 6)  # response coefficients
)

# Fit TLCQM model
Y_matched, beta_hat = fit_TLCQM(
    dat_source=dat_source,
    dat_target=dat_target,
    X_dat_tensor=None,
    n_sampler=3000,
    random_state=42,
    # Engression model hyperparameters
    eng_num_layer=2,
    eng_hidden_dim=100,
    eng_noise_dim=5,
    eng_lr=0.001,
    eng_num_epochs=1000,
    eng_pred_sample_size=500,
    # Quantile matching hyperparameters
    qm_stop_eps=1e-8,
    qm_max_iter=1000,
    qm_positive=False,
    qm_verbose=False
)

# Y_matched: calibrated responses for X_dat_tensor
# beta_hat: estimated quantile matching coefficients
```

### Running Experiments on HPC

```bash
# Run simulation experiment
sbatch Sim_TLCQM.sbatch

# Run apartment data experiment
sbatch Apartment_TLCQM.sbatch
```

## Data

The `data/` directory contains:
- `apartments_for_rent_classified_100K.csv`: Real-world apartment rental dataset for experiments

## Contribute

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License

TLCQM is MIT licensed, as found in the [LICENSE](LICENSE.md) file.
