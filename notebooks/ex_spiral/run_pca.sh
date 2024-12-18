#!/bin/bash
#SBATCH --job-name=spiral_pca
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
N=1000
d=250

srcpath=/data/gpfs/projects/punim0638/stephenz/QROT/notebooks/ex_spiral
save_mats=true

paramstr="--eps_quad_idx 16 --eps_ent_idx 2 --k_idx 10 --eps_epanech_idx 13 --eps_gauss_idx 9 --eps_gauss_l2_idx 16 --k_magic_idx 1"

echo $N $seed $d $dim $save_mats
echo $paramstr

julia spiral.jl --N $N --d $d --pca $dim --seed $seed --threads 4 --save_mats $save_mats $paramstr
