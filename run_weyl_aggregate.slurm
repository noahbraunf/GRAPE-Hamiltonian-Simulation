#!/bin/bash
#SBATCH --job-name=weyl_agg
#SBATCH --account=rmacc-general
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/weyl_agg_%j.out
#SBATCH --error=logs/weyl_agg_%j.err
#SBATCH --export=ALL

CONFIG="${CONFIG:-mediator}"
D_QUDIT="${D_QUDIT:-2}"

module purge
module load uv

source "$UV_ENVS/grape/bin/activate"

python weyl_sweep.py \
  --mode aggregate \
  --config "$CONFIG" \
  --d_qudit "$D_QUDIT" \
  --weyl_step_denom 8 \
  --T_min 0.05 --T_max 4.0 --n_T 20 \
  --output_dir "results/weyl"
