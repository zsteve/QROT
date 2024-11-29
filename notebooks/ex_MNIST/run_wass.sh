#!/bin/bash
#SBATCH --job-name=MNIST_wass
#SBATCH --output=output_%A_%a.out
#SBATCH --error=error_%A_%a.out
#SBATCH --array=1-10
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=gpu-a100-short
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1

srcpath=/data/gpfs/projects/punim0638/stephenz/QROT/notebooks/ex_MNIST
# SLURM_ARRAY_TASK_ID

source ~/.bashrc
conda activate qrot_py
# JULIA_NUM_THREADS=4 julia mnist_wass.jl --N 100 --seed $SLURM_ARRAY_TASK_ID --threads 4 --w 0 --save_mats false
# JULIA_NUM_THREADS=4 julia mnist_wass.jl --N 100 --seed $SLURM_ARRAY_TASK_ID --threads 4 --w 1 --save_mats false
# JULIA_NUM_THREADS=4 julia mnist_wass.jl --N 100 --seed $SLURM_ARRAY_TASK_ID --threads 4 --w 2 --save_mats false
JULIA_NUM_THREADS=4 julia mnist_wass.jl --N 100 --seed $SLURM_ARRAY_TASK_ID --threads 4 --w 3 --save_mats false
