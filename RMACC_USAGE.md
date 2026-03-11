# Running the Qudit SWAP Sweep on RMACC Alpine

## Prerequisites

- Active RMACC account with allocation on Alpine
- Access to the project repo on Alpine (clone or scp)

## 1. Connect to Alpine

```bash
ssh <username>@login.rc.colorado.edu
```

Once logged in, navigate to your scratch or projects directory:

```bash
cd /scratch/alpine/<username>
# or
cd /projects/<allocation>/<username>
```

## 2. Clone the repository

```bash
git clone <repo_url> GRAPE-Hamiltonian-Simulation
cd GRAPE-Hamiltonian-Simulation
```

## 3. Set up the Python environment (one-time)

CURC provides uv as a module — do not install it manually via curl.

Creating the virtualenv directory is a lightweight operation and can be done
on the login node. **Package installation (`uv pip install`) must run inside
an interactive job** because it downloads and compiles packages, which is
prohibited on login nodes.

```bash
# On the login node — create the venv directory only
module load uv
uv venv $UV_ENVS/grape
```

Then start an interactive compute job to install packages:

```bash
sinteractive --partition=amilan --ntasks=1 --cpus-per-task=2 \
             --mem=4G --time=01:00:00 --account=rmacc-general
```

Once the interactive shell opens on a compute node:

```bash
module load uv
source $UV_ENVS/grape/bin/activate
uv pip install "jax[cpu]" optax dynamiqs jaxtyping matplotlib
exit   # return to login node when done
```

`$UV_ENVS` is set automatically by `module load uv` to
`/projects/$USER/software/uv/envs`. The SLURM scripts activate this same
environment via `source $UV_ENVS/grape/bin/activate`.

## 4. Edit SLURM scripts — set your allocation code

Open `run_d2.slurm`, `run_d2_aggregate.slurm`, `run_d4.slurm`, and `run_d4_aggregate.slurm`
and replace:

```
#SBATCH --account=YOUR_PROJECT_CODE
```

with your actual Alpine allocation, e.g.:

```
#SBATCH --account=rmacc-general
```

Check your available allocations:

```bash
sacctmgr show assoc user=$USER format=account,partition
```

The scripts already include `#SBATCH --qos=normal`, which is required for the
`amilan` partition. Valid QoS options for `amilan`:

| QoS | Max walltime | Max jobs/user |
|-----|-------------|---------------|
| `normal` | 1 day | 1000 |
| `long` | 7 days | 200 |
| `testing` | 1 hour | 5 |

## 5. Verify the setup (optional — requires an interactive job)

The check mode triggers JAX JIT compilation, which is compute-intensive and
must not run on the login node. Start an interactive job first:

```bash
sinteractive --partition=amilan --ntasks=1 --cpus-per-task=2 \
             --mem=4G --time=00:30:00 --account=rmacc-general
```

Then inside the interactive shell:

```bash
module load uv
source $UV_ENVS/grape/bin/activate
python grape-curc-sim.py --mode check
exit
```

Expected output:

```
d=2: 27 generators, dim=8, max Hermitian err=0.00e+00
d=4: 27 generators, dim=16, max Hermitian err=0.00e+00
All checks passed.
```

## 6. Submit the array jobs

```bash
mkdir -p logs results/d2 results/d4

# d=2 sweep: 50 tasks × 4 hours each on amilan
sbatch run_d2.slurm

# d=4 sweep: 50 tasks × 8 hours each on amilan (larger matrices)
sbatch run_d4.slurm
```

Both sweeps run concurrently and independently. Each array task writes one file:

```
results/d2/d2_T0000.npz  …  results/d2/d2_T0049.npz
results/d4/d4_T0000.npz  …  results/d4/d4_T0049.npz
```

## 7. Monitor jobs

```bash
# List your running/pending jobs
squeue -u $USER

# Auto-refresh every 30 seconds
watch -n 30 squeue -u $USER

# Read the output of a specific task (e.g. task 5 of job 12345)
cat logs/d2_12345_5.out

# Count completed result files
ls results/d2/d2_T*.npz | wc -l   # should reach 50
ls results/d4/d4_T*.npz | wc -l
```

## 8. Aggregate results

Submit the aggregate job after the array job finishes. Use SLURM's dependency system
so it starts automatically:

```bash
# Capture job IDs to use with --dependency
D2_JID=$(sbatch --parsable run_d2.slurm)
D4_JID=$(sbatch --parsable run_d4.slurm)
echo "d=2 job: $D2_JID   d=4 job: $D4_JID"

# Then submit aggregate jobs with automatic dependency
sbatch --dependency=afterok:$D2_JID run_d2_aggregate.slurm
sbatch --dependency=afterok:$D4_JID run_d4_aggregate.slurm
```

Or submit manually once `squeue` shows no remaining array tasks:

```bash
sbatch run_d2_aggregate.slurm
sbatch run_d4_aggregate.slurm
```

This produces:

```
results/d2/d2_sweep.npz
results/d4/d4_sweep.npz
```

## 9. Download results to your laptop

```bash
# Run these commands from your local machine
# Adjust /scratch/alpine/<username> to /projects/<allocation>/<username> if needed
scp <username>@login.rc.colorado.edu:/scratch/alpine/<username>/GRAPE-Hamiltonian-Simulation/results/d2/d2_sweep.npz .
scp <username>@login.rc.colorado.edu:/scratch/alpine/<username>/GRAPE-Hamiltonian-Simulation/results/d4/d4_sweep.npz .
```

## 10. Load and inspect results

```python
import numpy as np

data = np.load("d2_sweep.npz", allow_pickle=True)
print(data.files)
# ['T_values', 'fidelities', 'best_omegas', 'all_loss_histories',
#  'generator_labels', 'd_qudit', 'M', 'n_generators', 'n_restarts',
#  'epochs', 'lr', 'T_min', 'T_max', 'n_T']

T      = data["T_values"]           # shape (50,)  — evolution times
F      = data["fidelities"]         # shape (50,)  — best fidelity per T
omegas = data["best_omegas"]        # shape (50, 40, 27) — optimal pulses per T
hist   = data["all_loss_histories"] # shape (50, 50, 4000) — all restarts per T

# Find the quantum speed limit (first T where fidelity exceeds threshold)
threshold = 0.99
idx = np.argmax(F > threshold)
if F[idx] > threshold:
    print(f"Speed limit (F > {threshold}): T = {T[idx]:.3f}")
else:
    print("Fidelity never exceeded threshold — extend T_max and re-run")
```

## Troubleshooting

**Module not found / import error**

Confirm the environment is activated. The SLURM scripts do this automatically, but if
running manually:
```bash
module load uv
source $UV_ENVS/grape/bin/activate
```

**Job fails immediately with a JAX error**

Check `logs/d2_<jobid>_<taskid>.err`. The most common cause is the wrong Python module.
Run `module avail python` on Alpine and update the `module load python/3.11` line in the
SLURM scripts to match an available version.

**Jobs sit in the queue for a long time**

Shorter walltime → higher scheduling priority. If the default 4 h / 8 h is too long for
your initial tests, reduce it (e.g. `--time=01:00:00`) for a small `--n_T` test run.
The `amilan` partition allows up to 24 hours.

**Some array tasks failed**

Check which `T_idx` files are missing:

```bash
for i in $(seq 0 49); do
  f="results/d2/d2_T$(printf '%04d' $i).npz"
  [[ -f "$f" ]] || echo "missing: T_idx=$i  ($f)"
done
```

Resubmit only the missing indices:

```bash
sbatch --array=3,7,12 run_d2.slurm
```

**Fidelity never reaches 1 — need a wider T range**

Run additional T values beyond the original range and aggregate everything together:

```bash
# Example: extend d=2 from T=12 to T=20 with 20 new points
# First, note that new T_idx values must not overlap with existing ones.
# Easiest approach: re-run with a wider range and higher n_T from scratch:
#   --T_min 0.3 --T_max 20.0 --n_T 70
# Then aggregate all 70 files.
```

**Aggregate raises "T mismatch" error**

This means some `.npz` files in `--output_dir` were produced with different sweep
parameters. Remove or move the stale files, then re-run aggregate.
