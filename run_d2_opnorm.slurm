#!/bin/bash
#SBATCH --job-name=grape_d2_op
#SBATCH --account=rmacc-general
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/d2_op_%A_%a.out
#SBATCH --error=logs/d2_op_%A_%a.err

module purge
module load uv

source "$UV_ENVS/grape/bin/activate"

export JAX_PLATFORM_NAME=cpu
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p results/d2_opnorm logs

python grape-curc-sim.py \
  --mode run \
  --d_qudit 2 \
  --normalize op_norm \
  --T_min 0.3 --T_max 12.0 --n_T 20 \
  --M 40 \
  --n_restarts 50 \
  --epochs 4000 \
  --lr 0.005 \
  --output_dir results/d2_opnorm
# SLURM_ARRAY_TASK_ID is read automatically inside the script
