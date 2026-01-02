#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang
Last Editing: Dec 31, 2024

Description: Synthesize the outputs from repeated experiments.
"""

import numpy as np
import pandas as pd

#=======================================================================================#

B = 1000
res_all = pd.DataFrame()
for b in range(1, B+1):
    res_tmp = pd.read_csv('./Results/Simulation_Concept_Covariate_'+str(b)+'.csv')
    res_all = pd.concat([res_all, res_tmp], axis=0)
res_all = res_all.reset_index(drop=True)
res_all.to_csv('./Results_Syn/Simulation_Concept_Covariate_TLCQM.csv', index=False)
    
    