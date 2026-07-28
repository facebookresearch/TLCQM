#!/bin/bash
#SBATCH --job-name sim_tlcqm_ratio_abla_cond
#SBATCH --partition short
#SBATCH --array=1-1000
#SBATCH --time 0-11:10
#SBATCH --mem-per-cpu=25000
#SBATCH -o out/sim_tlcqm_ratio_abla_cond_%j.out
#SBATCH -e out/sim_tlcqm_ratio_abla_cond_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yikun@uw.edu

module load Python
source ../yikun_venv/bin/activate
i=${SLURM_ARRAY_TASK_ID}
python ./Sim_TLCQM_Ratio_Abla_Conditional.py ${i}
