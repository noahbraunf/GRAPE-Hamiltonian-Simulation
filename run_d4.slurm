#!/bin/bash
#SBATCH --job-name=grape_d4
#SBATCH --account=rmacc-general
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-49
#SBATCH --output=logs/d4_%A_%a.out
#SBATCH --error=logs/d4_%A_%a.err

module purge
module load uv

source $UV_ENVS/grape/bin/activate

export JAX_PLATFORM_NAME=cpu
export XLA_FLAGS="--xla_cpu_multi_thread_eigen_intra_op_parallelism=1"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p results/d4 logs

python grape-curc-sim.py \
    --mode run \
    --d_qudit 4 \
    --T_min 0.3 --T_max 15.0 --n_T 50 \
    --M 40 \
    --n_restarts 50 \
    --epochs 4000 \
    --lr 0.005 \
    --output_dir results/d4
