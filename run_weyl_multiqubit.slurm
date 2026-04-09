#!/bin/bash
#SBATCH --job-name=weyl_mq2
#SBATCH --account=rmacc-general
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-199%5
#SBATCH --output=logs/weyl_mq_%A_%a.out
#SBATCH --error=logs/weyl_mq_%A_%a.err
#SBATCH --export=ALL

module purge
module --ignore_cache load uv

source "$UV_ENVS/grape/bin/activate"

export JAX_PLATFORM_NAME=cpu
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p results/weyl logs

python weyl_sweep.py \
  --mode run \
  --config multiqubit_2 \
  --d_qudit 4 \
  --weyl_step_denom 8 \
  --T_min 0.05 --T_max 6.0 --n_T 20 \
  --M 40 \
  --n_restarts 50 \
  --epochs 4000 \
  --lr 0.005 \
  --output_dir "results/weyl"
