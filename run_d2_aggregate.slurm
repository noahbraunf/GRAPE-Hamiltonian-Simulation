#!/bin/bash
#SBATCH --job-name=grape_d2_agg
#SBATCH --account=rmacc-general
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/d2_agg_%j.out
#SBATCH --error=logs/d2_agg_%j.err

module purge
module load uv

source "$UV_ENVS/grape/bin/activate"

# --- frobenius-norm results ---
python grape-curc-sim.py \
  --mode aggregate \
  --d_qudit 2 \
  --T_min 0.3 --T_max 12.0 --n_T 20 \
  --n_restarts 50 \
  --epochs 4000 \
  --output_dir results/d2_frobnorm

python grape-curc-sim.py \
  --mode analyze \
  --d_qudit 2 \
  --output_dir results/d2_frobnorm

# --- trotter+cosine results ---
python grape-curc-sim.py \
  --mode aggregate \
  --d_qudit 2 \
  --T_min 0.3 --T_max 12.0 --n_T 20 \
  --n_restarts 50 \
  --epochs 4000 \
  --output_dir results/d2_trotter

python grape-curc-sim.py \
  --mode analyze \
  --d_qudit 2 \
  --output_dir results/d2_trotter
