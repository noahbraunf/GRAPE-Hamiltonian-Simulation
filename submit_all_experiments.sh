#!/bin/bash
set -euo pipefail

# Submit d1 (direct, no ancilla) frobnorm job.
d1_frobnorm_jobid=$(sbatch --parsable run_d1_frobnorm.slurm)

# Aggregate d1 after compute job succeeds.
d1_aggregate_jobid=$(sbatch --parsable --dependency=afterok:${d1_frobnorm_jobid} run_d1_aggregate.slurm)

# Submit all d2 compute jobs first.
d2_base_jobid=$(sbatch --parsable run_d2.slurm)
d2_frobnorm_jobid=$(sbatch --parsable run_d2_frobnorm.slurm)
d2_trotter_jobid=$(sbatch --parsable run_d2_trotter.slurm)

# Aggregate d2 only after all d2 compute jobs succeed.
d2_dependencies="${d2_base_jobid}:${d2_frobnorm_jobid}:${d2_trotter_jobid}"
d2_aggregate_jobid=$(sbatch --parsable --dependency=afterok:${d2_dependencies} run_d2_aggregate.slurm)

# Submit all d4 compute jobs first.
d4_base_jobid=$(sbatch --parsable run_d4.slurm)
d4_frobnorm_jobid=$(sbatch --parsable run_d4_frobnorm.slurm)
d4_trotter_jobid=$(sbatch --parsable run_d4_trotter.slurm)

# Aggregate d4 only after all d4 compute jobs succeed.
d4_dependencies="${d4_base_jobid}:${d4_frobnorm_jobid}:${d4_trotter_jobid}"
d4_aggregate_jobid=$(sbatch --parsable --dependency=afterok:${d4_dependencies} run_d4_aggregate.slurm)

echo "Submitted d1 jobs:"
echo "  run_d1_frobnorm.slurm   -> ${d1_frobnorm_jobid}"
echo "  run_d1_aggregate.slurm  -> ${d1_aggregate_jobid} (afterok:${d1_frobnorm_jobid})"
echo
echo "Submitted d2 jobs:"
echo "  run_d2.slurm            -> ${d2_base_jobid}"
echo "  run_d2_frobnorm.slurm   -> ${d2_frobnorm_jobid}"
echo "  run_d2_trotter.slurm    -> ${d2_trotter_jobid}"
echo "  run_d2_aggregate.slurm  -> ${d2_aggregate_jobid} (afterok:${d2_dependencies})"
echo
echo "Submitted d4 jobs:"
echo "  run_d4.slurm            -> ${d4_base_jobid}"
echo "  run_d4_frobnorm.slurm   -> ${d4_frobnorm_jobid}"
echo "  run_d4_trotter.slurm    -> ${d4_trotter_jobid}"
echo "  run_d4_aggregate.slurm  -> ${d4_aggregate_jobid} (afterok:${d4_dependencies})"
