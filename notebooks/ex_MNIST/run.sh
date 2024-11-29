#!/bin/bash
#SBATCH --job-name=MNIST
#SBATCH --output=output_%A_%a.out
#SBATCH --error=error_%A_%a.out
#SBATCH --array=1-10
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=cascade
#SBATCH --time=03:00:00

srcpath=/data/gpfs/projects/punim0638/stephenz/QROT/notebooks/ex_MNIST
# SLURM_ARRAY_TASK_ID

source ~/.bashrc
conda activate qrot_py
julia mnist.jl --N 250 --seed $SLURM_ARRAY_TASK_ID --threads 4
