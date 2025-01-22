#!/bin/bash
#SBATCH --job-name=activeset
#SBATCH --output=output_%A_%a.out
#SBATCH --error=error_%A_%a.out
#SBATCH --array=31-50
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=mig
#SBATCH --time=08:00:00

parameters=`sed -n "${SLURM_ARRAY_TASK_ID} p" params`
parameterArray=($parameters)

seed=${parameterArray[0]}
N=${parameterArray[1]}

srcpath=/data/gpfs/projects/punim0638/stephenz/QROT/notebooks/ex_algs

echo $seed $N

skipdense=false
if (( $N > 25000)); then
    skipdense=true
fi
echo skipdense=$skipdense

julia activeset.jl --N $N --seed $seed --skipdense $skipdense --d 250
