#!/bin/bash
#SBATCH --job-name=grape_d2_agg
#SBATCH --account=rmacc-general
#SBATCH --partition=amilan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/d2_agg_%j.out
#SBATCH --error=logs/d2_agg_%j.err

module purge
module load python/3.11

source .venv/bin/activate

python grape-curc-sim.py \
  --mode aggregate \
  --d_qudit 2 \
  --T_min 0.3 --T_max 12.0 --n_T 50 \
  --n_restarts 50 \
  --epochs 4000 \
  --output_dir results/d2
