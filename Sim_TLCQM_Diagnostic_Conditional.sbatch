#!/bin/bash
#SBATCH --job-name sim_tlcqm_diag_cond
#SBATCH --partition short
#SBATCH --array=1-500
#SBATCH --time 0-11:10
#SBATCH --mem-per-cpu=20000
#SBATCH -o out/sim_tlcqm_diag_cond_%j.out
#SBATCH -e out/sim_tlcqm_diag_cond_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yikun@uw.edu

module load Python
source ../yikun_venv/bin/activate
i=${SLURM_ARRAY_TASK_ID}
python ./Sim_TLCQM_Diagnostic_Conditional.py ${i}
