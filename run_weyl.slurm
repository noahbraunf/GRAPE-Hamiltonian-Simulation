#!/bin/bash
#SBATCH --job-name=weyl_sweep
#SBATCH --account=rmacc-general
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-199%10
#SBATCH --output=logs/weyl_%A_%a.out
#SBATCH --error=logs/weyl_%A_%a.err
#SBATCH --export=ALL

# ── Set via: sbatch --export=ALL,CONFIG=mediator,D_QUDIT=2 run_weyl.slurm ──
CONFIG="${CONFIG:-mediator}"
D_QUDIT="${D_QUDIT:-2}"

module purge
module --ignore_cache load uv

source "$UV_ENVS/grape/bin/activate"

export JAX_PLATFORM_NAME=cpu
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p results/weyl logs

python weyl_sweep.py \
  --mode run \
  --config "$CONFIG" \
  --d_qudit "$D_QUDIT" \
  --weyl_step_denom 8 \
  --T_min 0.05 --T_max 4.0 --n_T 20 \
  --M 40 \
  --n_restarts 50 \
  --epochs 4000 \
  --lr 0.005 \
  --output_dir "results/weyl"
