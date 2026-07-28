# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
@author: Yikun Zhang
Last Editing: Jul 24, 2026

Description: Synthesize repeated conditional TLCQM experiments.
"""

import pandas as pd

#=======================================================================================#


def synthesize(file_name, B, output_name):
    res_all = []
    for b in range(1, B + 1):
        res_all.append(pd.read_csv(file_name.format(b)))
    pd.concat(res_all, axis=0, ignore_index=True).to_csv(
        output_name, index=False
    )


def synthesize_pair(file_name1, file_name2, B, output_name):
    res_all = []
    for b in range(1, B + 1):
        res_all.append(pd.read_csv(file_name1.format(b)))
        res_all.append(pd.read_csv(file_name2.format(b)))
    pd.concat(res_all, axis=0, ignore_index=True).to_csv(
        output_name, index=False
    )


synthesize(
    "./Results/Simulation_Concept_Covariate_{}_conditional_mean.csv",
    1000,
    "./Results_Syn/Simulation_Concept_Covariate_TLCQM_Conditional_Mean.csv",
)

synthesize(
    "./Results/Simulation_Concept_Covariate_{}_conditional_mean_abla.csv",
    1000,
    "./Results_Syn/Simulation_Concept_Covariate_TLCQM_Conditional_Mean_Abla.csv",
)

synthesize(
    "./Results/Simulation_Diagnostic_{}_conditional_mean.csv",
    500,
    "./Results_Syn/Simulation_TLCQM_Conditional_Mean_Diagnostic.csv",
)

synthesize(
    "./Results/Apartment_{}_conditional_mean.csv",
    500,
    "./Results_Syn/Apartment_TLCQM_Conditional_Mean.csv",
)

synthesize_pair(
    "./Results/Apartment_{}_conditional_mean.csv",
    "./Results/Apartment_{}_baselines_extended.csv",
    500,
    "./Results_Syn/Apartment_TLCQM_Conditional_Mean_Baselines.csv",
)

synthesize_pair(
    "./Results/Apartment_{}_conditional_mean.csv",
    "./Results/Apartment_{}_Compare_Extended.csv",
    500,
    "./Results_Syn/Apartment_TLCQM_Conditional_Mean_Compare.csv",
)

synthesize(
    "./Results/Simulation_Concept_Covariate_{}_conditional_mean_grid.csv",
    1000,
    "./Results_Syn/Simulation_Concept_Covariate_TLCQM_Conditional_Mean_Grid.csv",
)
