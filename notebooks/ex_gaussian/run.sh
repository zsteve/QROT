#!/bin/bash
#SBATCH --job-name=gaussian
#SBATCH --output=output_%A_%a.out
#SBATCH --error=error_%A_%a.out
#SBATCH --array=1-90
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=mig
#SBATCH --time=08:00:00

parameters=`sed -n "${SLURM_ARRAY_TASK_ID} p" params`
parameterArray=($parameters)

seed=${parameterArray[0]}
d=${parameterArray[1]}
N=${parameterArray[2]}

srcpath=/data/gpfs/projects/punim0638/stephenz/QROT/notebooks/ex_gaussian

if [[ $seed -eq 1 ]]; then
    save_mats=true
else
    save_mats=false
fi

echo $N $seed $save_mats

julia gaussian.jl --N $N --d $d --seed $seed --threads 4 --save_mats $save_mats
