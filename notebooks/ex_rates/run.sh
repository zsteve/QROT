#!/bin/bash
#SBATCH --job-name=rates
#SBATCH --output=output_%A_%a.out
#SBATCH --error=error_%A_%a.out
#SBATCH --array=1-175
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=mig
#SBATCH --time=01:00:00

parameters=`sed -n "${SLURM_ARRAY_TASK_ID} p" params`
parameterArray=($parameters)

N=${parameterArray[0]}
i=${parameterArray[1]}

srcpath=/data/gpfs/projects/punim0638/stephenz/QROT/notebooks/ex_rates

echo $N $i

JULIA_NUM_THREADS=4 julia rates.jl --N $N --i $i
