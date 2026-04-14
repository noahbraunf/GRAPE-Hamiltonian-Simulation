#!/bin/bash
set -euo pipefail

module purge
module load slurm/alpine

# Submit d1 (direct, no ancilla) frobnorm job.
d1_jobid=$(sbatch --parsable run_d1_frobnorm.slurm)
d1_agg=$(sbatch --parsable --dependency=afterok:${d1_jobid} run_d1_aggregate.slurm)

# Submit d2 frobnorm job.
d2_jobid=$(sbatch --parsable run_d2_frobnorm.slurm)
d2_agg=$(sbatch --parsable --dependency=afterok:${d2_jobid} run_d2_aggregate.slurm)

# Submit d4 frobnorm job.
d4_jobid=$(sbatch --parsable run_d4_frobnorm.slurm)
d4_agg=$(sbatch --parsable --dependency=afterok:${d4_jobid} run_d4_aggregate.slurm)

echo "Submitted jobs:"
echo "  d1 frobnorm: ${d1_jobid}  ->  aggregate: ${d1_agg}"
echo "  d2 frobnorm: ${d2_jobid}  ->  aggregate: ${d2_agg}"
echo "  d4 frobnorm: ${d4_jobid}  ->  aggregate: ${d4_agg}"
