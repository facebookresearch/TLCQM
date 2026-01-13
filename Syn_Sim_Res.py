#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description: Synthesize the outputs from repeated experiments.
"""

import numpy as np
import pandas as pd

#=======================================================================================#

# B = 1000
# res_all = pd.DataFrame()
# for b in range(1, B+1):
#     res_tmp = pd.read_csv('./Results/Simulation_Concept_Covariate_'+str(b)+'_new.csv')
#     res_all = pd.concat([res_all, res_tmp], axis=0)
# res_all = res_all.reset_index(drop=True)
# res_all.to_csv('./Results_Syn/Simulation_Concept_Covariate_TLCQM_new.csv', index=False)

B = 1000
res_all = pd.DataFrame()
for b in range(1, B+1):
    res_tmp = pd.read_csv('./Results/Simulation_Concept_Covariate_'+str(b)+'_new2.csv')
    res_all = pd.concat([res_all, res_tmp], axis=0)
res_all = res_all.reset_index(drop=True)
res_all.to_csv('./Results_Syn/Simulation_Concept_Covariate_TLCQM_new2.csv', index=False)    

# res_all = pd.DataFrame()
# for b in range(1, B+1):
#     res_tmp = pd.read_csv('./Results/Simulation_Concept_Covariate_'+str(b)+'_Compare.csv')
#     res_all = pd.concat([res_all, res_tmp], axis=0)
# res_all = res_all.reset_index(drop=True)
# res_all.to_csv('./Results_Syn/Simulation_Concept_Covariate_TLCQM_Compare.csv', index=False)

res_all = pd.DataFrame()
for b in range(1, B+1):
    res_tmp = pd.read_csv('./Results/Simulation_Concept_Covariate_'+str(b)+'_Compare_new2.csv')
    res_all = pd.concat([res_all, res_tmp], axis=0)
res_all = res_all.reset_index(drop=True)
res_all.to_csv('./Results_Syn/Simulation_Concept_Covariate_TLCQM_Compare_new2.csv', index=False)


B = 500
res_full = pd.DataFrame()
for b in range(1, B+1):
    res_tmp = pd.read_csv('./Results/Apartment_'+str(b)+'_new.csv')
    res_tmp2 = pd.read_csv('./Results/Apartment_'+str(b)+'_Compare_new.csv')
    res_full = pd.concat([res_full, res_tmp, res_tmp2], axis=0)
res_full = res_full.reset_index(drop=True)
res_full.to_csv('./Results_Syn/Apartment_TLCQM_Results_new.csv', index=False)