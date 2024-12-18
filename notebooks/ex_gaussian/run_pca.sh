#!/bin/bash
#SBATCH --job-name=gaussian_pca
#SBATCH --output=output_%A_%a.out
#SBATCH --error=error_%A_%a.out
#SBATCH --array=1-500
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=mig
#SBATCH --time=01:00:00

parameters=`sed -n "${SLURM_ARRAY_TASK_ID} p" params_pca`
parameterArray=($parameters)

seed=${parameterArray[0]}
dim=${parameterArray[1]}
N=500
d=250

srcpath=/data/gpfs/projects/punim0638/stephenz/QROT/notebooks/ex_gaussian
save_mats=false

# 250
# eps_quad_idx=16
# eps_ent_idx=8
# k_idx=23

# 500
eps_quad_idx=17
eps_ent_idx=7
k_idx=23

echo $N $seed $d $dim $save_mats
echo $eps_quad_idx $eps_ent_idx $k_idx

julia gaussian.jl --N $N --d $d --pca $dim --seed $seed --threads 4 --save_mats $save_mats --eps_quad_idx $eps_quad_idx --eps_ent_idx $eps_ent_idx --k_idx $k_idx 
