#!/bin/bash
#SBATCH --job-name=grape_d4_agg
#SBATCH --account=rmacc-general
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/d4_agg_%j.out
#SBATCH --error=logs/d4_agg_%j.err

module purge
module --ignore_cache load uv

source "$UV_ENVS/grape/bin/activate"

# --- frobenius-norm results ---
python grape-curc-sim.py \
  --mode aggregate \
  --d_qudit 4 \
  --T_min 0.3 --T_max 15.0 --n_T 20 \
  --n_restarts 50 \
  --epochs 4000 \
  --output_dir results/d4_frobnorm

python grape-curc-sim.py \
  --mode analyze \
  --d_qudit 4 \
  --output_dir results/d4_frobnorm

# --- trotter+cosine results ---
python grape-curc-sim.py \
  --mode aggregate \
  --d_qudit 4 \
  --T_min 0.3 --T_max 15.0 --n_T 20 \
  --n_restarts 50 \
  --epochs 4000 \
  --output_dir results/d4_trotter

python grape-curc-sim.py \
  --mode analyze \
  --d_qudit 4 \
  --output_dir results/d4_trotter
