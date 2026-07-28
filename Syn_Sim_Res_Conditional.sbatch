#!/bin/bash
#SBATCH --job-name syn_tlcqm_cond
#SBATCH --partition short
#SBATCH --ntasks 1
#SBATCH --time 0-10:55
#SBATCH --mem-per-cpu=7000
#SBATCH -o out/syn_tlcqm_cond.out
#SBATCH -e out/syn_tlcqm_cond.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yikun@uw.edu

module load Python
source ../yikun_venv/bin/activate
python ./Syn_Sim_Res_Conditional.py
