# Running CORAL/GRL GWHD experiments on the FSE CSlab HPC cluster

Based on `hpcAccess/HPC-info.pdf`, `slurm_documentation.pdf`, and `EESSI_documentation.pdf`, plus this repo's
existing training scripts (`train_GWHD_coralfrcnn.py`, `train_GWHD_dgfrcnn.py`) and RunPod run scripts
(`run_gwhd_dg.sh`, `run_gwhd_baseline.sh`).

## 1. Account + access (one-time)

1. Activate your account at `https://account.fse-cslab.nl/activate/` (must be on EDUROAM/UMnet), select
   NL -> Maastricht University, log in with student number, accept the AUP.
2. Generate an ed25519 SSH key **on your own laptop** (not on the cluster), passphrase >= 15 chars:
   `ssh-keygen -t ed25519 -C "your_i_number"`
3. At `https://account.fse-cslab.nl/chsh/`, set login shell to `/bin/bash` and paste your **public** key.
4. Connect: `ssh i6356965@dacsgpu0001.fse-cslab.nl` (your directory identity is
   `uid=i6356965,ou=students,ou=LocalUsers,dc=fse-cslab,dc=nl` — keep that string handy for admin support
   requests. Note: the doc says `dacsgpu0001`, the cluster spec table calls it `dacsgpu001` — try the
   login-node name from your welcome email if one doesn't resolve).
5. Off-campus: set up the VPN first (link in HPC-info.pdf). When picking the VPN group, use **07**, not
   06-AssignedStudents (that group is only for downloading the client).

Cluster shape you're working with: 1 login node + 2 further GPU worker nodes, each with 4x L40 (48GB) GPUs,
32 cores/64 threads, 768GB RAM, plus a CPU-only worker (32 cores/256GB). One partition. Since this is shared
with other students, request only what a run actually needs (see §3).

## 2. Environment setup (one-time, on the login node)

The cluster doesn't provide a pre-built conda — you install your own Miniconda, matching the `DGOD` conda-env
convention already used by `run_gwhd_dg.sh` / `run_gwhd_baseline.sh` on RunPod. `setup_env.sh` in this folder
does the whole thing:

```bash
cd ~/newThesisRepo          # wherever you clone/copy the repo on the cluster
bash hpc/setup_env.sh
```

It installs Miniconda into `/data/i6356965/miniconda3` (persistent, not subject to a tight home quota),
symlinks `~/miniconda3` to it (so the existing `run_gwhd_dg.sh`/`run_gwhd_baseline.sh` conda-detection logic
works unmodified), creates a `DGOD` env with Python 3.12 (matching what your RunPod logs show), and installs
`requirements.txt`. Re-running it is safe — it skips steps that are already done.

Add this once so conda is available in every new shell:

```bash
echo 'source ~/miniconda3/etc/profile.d/conda.sh' >> ~/.bashrc
```

PyPI's `torch`/`torchvision` wheels (installed via `requirements.txt`) bundle their own CUDA runtime, so you
don't need to `module load` a separate CUDA module for training — just make sure the driver on the worker node
supports it (L40s are fine with any current torch 2.x build). To check: `srun --gpus=1 --pty nvidia-smi` on a
worker node, or just look at the `nvidia_smi.txt` snapshot any `hpc/*.slurm` job writes into its run dir.

## 3. Data placement

The GWHD dataset (`newThesisRepo/data/gwhd_2021`, ~9.7GB) should **not** live in your home directory if home
has a small quota (typical on HPC) — same reasoning as putting Miniconda in `/data` above. Put it in `/data`
(mentioned in HPC-info.pdf as the shared file area) instead, e.g.:

```bash
mkdir -p /data/i6356965/gwhd_thesis
# clone/copy the repo code here, or symlink data/checkpoints from your home clone into /data
```

Check your actual home quota (`quota` or `df -h ~`) before assuming — if it's generous, this step is optional.

## 4. A important gotcha: PyTorch Lightning + SLURM

Your existing run logs (e.g. `coral_runs/bs8run01/coral-bs8-lr1e4-2.log`) show this warning:

```
The `srun` command is available on your system but is not used. HINT: If your intention is to run
Lightning on SLURM, prepend your python command with `srun` like so: srun python train_GWHD_coralfrcnn.py ...
```

Lightning auto-detects a SLURM allocation from env vars set by `sbatch` and switches to its `SLURMEnvironment`
plugin. If you then don't launch the actual training process with `srun`, it can misbehave (hang waiting for a
handshake, or mis-set devices). **Always launch training with `srun python ...` inside the sbatch script**, not
plain `python ...` — the templates below already do this.

## 5. Submitting a job

`train_coral.slurm` is a template mirroring `run_gwhd_dg.sh`/`run_gwhd_baseline.sh` but adapted for SLURM
instead of a persistent conda env + RunPod. Edit the `EXP`, `WEIGHTS_FILE`, and `--reg_weights` values, then:

```bash
sbatch hpc/train_coral.slurm coral run01           # exp=coral, tag=run01
sbatch hpc/train_coral.slurm non_dg baseline_run01  # baseline
```

Check status with `squeue -u $USER`, cancel with `scancel <JOBID>`, tail live output with:

```bash
JOB_ID=$(sbatch --parsable hpc/train_coral.slurm coral run01)
tail -f runs/gwhd_coral_run01/logs/stdout-*-$JOB_ID.out
```

Run `hpc/smoke_test.slurm` first on a new environment — it does a 1-epoch, small-batch dry run (reusing the
`smoke_GWHD` checkpoints pattern already in this repo) to confirm the venv, CUDA, and data path all work before
you burn a multi-hour allocation on a broken environment.

## 6. Resource sizing rationale

- `--gpus=1`: your training script (`train_GWHD_coralfrcnn.py`) runs single-GPU Lightning, batch_size default
  2 — one L40 (48GB) is far more than enough headroom, no reason to request more per job.
- `--cpus-per-task=16`: matches `--num_workers=16` default in the training script (1 CPU core per dataloader
  worker, plus the main process).
- `--mem=32G`: conservative for image + albumentations augmentation pipeline on GWHD; bump if you see OOM-kill
  in `stderr-*.err`.
- `--time`: `24:00:00` in the repro scripts — comfortable single-shot budget confirmed against `scontrol show
  partition research` (`MaxTime=7-00:00:00`). `train_coral.slurm` (`12:00:00`) and `smoke_test.slurm`
  (`00:15:00`) are also well inside that cap, no changes needed there.
- `--partition=research --account=research`: went through two wrong guesses before landing here, worth
  recording so it isn't relitigated. (1) `--partition=education --account=education`, the docs' literal
  example — wrong account, `scontrol show assoc_mgr users=$USER` showed `DefAccount=research` and no personal
  association under `education` at all. (2) `--partition=education --account=research` — account now correct,
  but `scontrol show partition education` revealed `AllowAccounts=education`: that partition flatly rejects
  any account other than `education`, so there was never a working combination through it for this user. (3)
  `--partition=research --account=research` — `scontrol show partition research` confirms
  `AllowAccounts=research,project`, so this is the account's actual home partition. Confirmed working.
  `sacctmgr` itself is unreachable from the login node on this cluster (connection refused to `slurmdbd`), so
  `scontrol show assoc_mgr users=$USER` / `scontrol show partition <name>` are the way to check any of this,
  not `sacctmgr`.

## 7. Running multiple experiments in parallel

There are up to ~12 GPUs total across worker nodes. Since each run only needs 1 GPU, submit each experiment
(baseline / GRL / CORAL, or a CORAL `--reg_weights` sweep) as a **separate `sbatch` call** rather than
requesting multiple GPUs in one job — SLURM will schedule them across whatever L40s are free:

```bash
for tag in run01 run02 run03; do
  sbatch hpc/train_coral.slurm coral "$tag"
done
```

For a systematic sweep (e.g. over `--reg_weights` or `--sampler`), a SLURM job array is cleaner than a bash
loop — ask if you want a template for that once you know which axis you're sweeping.

## 8. Reproduction attempt: baseline vs. DGKarthik (Aug 2026)

Context (from Petra Bosilj's email fwd. Mattia Dutto, 2026-08-12, "DGOD Code"): Mattia reproduced the paper's
expected ordering (baseline < DGKarthik) with BS=8, LR=1e-5, alpha_4=0.055, seeds 0/1/2, fully deterministic —
avg test mAP@0.5 of 54.7 (baseline) vs. 60.3 (DGKarthik). Crucially, **his baseline is a clean/stock torchvision
Faster R-CNN, not Karthik's `non_dg` path** (which reuses the repo's customized `fasterrcnn.py`/FastWILDS
detector — per-image RPN/ROI losses instead of per-batch — even with the DA heads switched off). That
implementation difference is the leading suspect for why the earlier `grlbaselineruns`/`grlbaseline1e-4` runs in
this repo came out statistically tied instead of showing the paper's gap.

**Two scripts, mirroring that split:**

- `train_GWHD_baseline_clean.py` — new. Stock `torchvision.models.detection.fasterrcnn_resnet50_fpn`, no
  `fasterrcnn.py` involved. Same `WheatDataset`/transforms/collate as Mattia's script, same AdamW(wd=5e-4) +
  `ReduceLROnPlateau` + `EarlyStopping(patience=10)` training loop, so the only thing that differs from the
  DGKarthik run is the detector implementation itself.
- `train_GWHD_dgfrcnn_mattia.py` — Mattia's forwarded script, used as-is for `--exp dg`. Two bugs fixed while
  wiring it up for SLURM (see the file's git diff): `from tqdm.notebook import tqdm` would fail outside a
  Jupyter kernel (needs `ipywidgets`), changed to plain `tqdm`; the final val/test evaluation had a hardcoded
  `ckpt_path=f"GWHD/{args.model_checkpoint}.ckpt"` that only worked if `--weights_folder` was left at its
  default and a redundant `--model-checkpoint` flag was passed — changed to resolve from the actual
  `--weights_folder`/`--weights_file` the run used, so parallel per-seed runs into distinct run dirs each
  evaluate their own checkpoint. Whitespace was already all-spaces (no tab/space fix needed on top of what
  Mattia already sent). Both scripts include the determinism block from Mattia's email verbatim (`random`/
  `numpy`/`pytorch_lightning`/`torch` seeding + `cudnn.deterministic`/`use_deterministic_algorithms`), gated
  behind `--deterministic`.

**Open assumption — flag before trusting the numbers:** the email only specifies `alpha_4 = 0.055`
("the only parameter of note"). The other four `reg_weights` (image/instance/consistency/cls weights) aren't
given. `hpc/repro_dgkarthik.slurm` uses this repo's existing convention (`0.5 0.5 0.5 <alpha_4> 0.0001`, same as
`run_gwhd_dg.sh`) with only the 4th value swapped to 0.055. If Mattia confirms different values for the other
four, edit `REG_WEIGHTS` at the top of that script before submitting.

**Running it — walltime and the checkpoint/resume workflow:**

These scripts run on the `research` partition (`--account=research`; see the §6 note on why
`education`/`education` and `education`/`research` both turned out to be dead ends). `scontrol show partition
research` confirms `MaxTime=7-00:00:00`, so both scripts request a comfortable single-shot `--time=24:00:00` —
no need to babysit them across multiple submissions the way `education`'s 2h cap would have required.

They still checkpoint every epoch and auto-resume, kept as a safety net rather than a load-bearing requirement:

- `ModelCheckpoint(save_last=True)` writes `runs/.../checkpoints/last.ckpt` every epoch; the script auto-resumes
  from it at startup if present — full trainer state (weights, optimizer, LR scheduler, epoch count,
  early-stopping patience counter). Covers node reboots, preemption, or a future stricter time limit.
- A `<weights_file>.done` sentinel file is written only after the final val/test evaluation genuinely
  completes. The "already exists, skip" guard checks this instead of the checkpoint itself (which legitimately
  exists mid-run) — so re-submitting a seed that's still `.done`-less resumes it, and re-submitting one that's
  already `.done` just skips it cleanly instead of erroring.

**In practice, just submit once per experiment:**

```bash
sbatch hpc/repro_baseline.slurm    # 3 jobs (seeds 0,1,2), lr=1e-5, BS=8, deterministic
sbatch hpc/repro_dgkarthik.slurm   # 3 jobs (seeds 0,1,2), lr=1e-5, BS=8, alpha_4=0.055, deterministic
```

If a run ever does get interrupted before finishing, the exact same command resumes it — no separate recovery
step needed.

Each `sbatch` call is a job array (`--array=0-2`), so all 3 seeds per experiment run concurrently across
whatever L40s are free. Results land in `runs/gwhd_repro_baseline_lr1e-5/` and `runs/gwhd_repro_dgkarthik_lr1e-5/`,
each with per-seed `checkpoints/` (`last.ckpt`, the best-`val_acc` checkpoint, and eventually `.done`),
`logs/train_seed<N>.log`, and a `config_snapshot/`. A seed is fully finished once its `.done` file exists (or
equivalently, once `TEST:` and a `val_acc` line appear near the end of its log) — average the 3 seeds' test
`val_acc` and compare against Mattia's 54.7/60.3.

One known gap in the resume logic for `train_GWHD_dgfrcnn_mattia.py` specifically: its `training_step` cycles
through an internal `self.mode` (0→1→2→3) each batch, which is a plain Python int, not part of the checkpointed
state. A resumed run always restarts at `mode=0` regardless of where it was cut off — harmless (just repeats up
to 3 extra sub-steps of that batch), not a correctness issue.

`train_coral.slurm` (CORAL runs) already targets `--partition=research --account=research` and requests
`--time=12:00:00`, comfortably inside `research`'s 7-day cap — no `PartitionTimeLimit`/`PartitionConfig` issue
expected there. It doesn't have the checkpoint/resume/`.done` logic the two repro scripts have, since it hasn't
needed it; say the word if you want that added too (e.g. as extra insurance for very long CORAL sweeps).

Mattia's email also mentions LR=1e-4 was tested (Sheet3 rows 3+5) but doesn't give numbers for it. Both scripts
take `lr` as an optional first argument if you want to run that comparison too:

```bash
sbatch hpc/repro_baseline.slurm 1e-4
sbatch hpc/repro_dgkarthik.slurm 1e-4
```

**RunPod alternative:** the `research` partition queue turned out to have ~250 jobs from another user ahead of
these (see the priority/queue discussion — flat `PRIORITY=1` on everything, so effectively FIFO by submission
order, and both GPU worker nodes were already fully occupied by that user's running jobs). If the cluster queue
isn't moving, `../run_repro_baseline.sh` and `../run_repro_dgkarthik.sh` in the repo root run the identical
seeds/hyperparameters on a RunPod pod instead — same conda-detection convention as `run_gwhd_dg.sh`/
`run_gwhd_baseline.sh`, same `runs/gwhd_repro_*_lr<LR>/` layout, so results from either source are directly
comparable. Seeds run sequentially by default (a single pod GPU may not fit 3 concurrent BS=8 trainings) and
still checkpoint/resume via the same `last.ckpt`/`.done` mechanism, which matters more here than on the cluster
if you're using an interruptible/spot RunPod instance. Unlike `sbatch`, these are plain foreground bash
processes — run them inside `tmux`/`screen` (or `nohup`) so they survive an SSH disconnect.

**`torch.OutOfMemoryError` on a 24GB pod GPU (RTX 4090 / A4500 / etc.):** real, not a bug — BS=8 at 1024x1024 is
a genuinely tight fit on a 24GB card (the university cluster's L40s have 48GB, so this hadn't come up there).
Both training scripts now accept `--batch_size` (was hardcoded to 8 in `train_GWHD_dgfrcnn_mattia.py`, now
overridable) and `--accumulate_grad_batches` (Lightning gradient accumulation), and both `run_repro_*.sh`
already export `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (a free fix specifically for allocator
fragmentation — check whether the OOM message's "reserved but unallocated" figure is large; if so this alone
may fix it). If the physical batch itself is just too big for the card, split it via env vars without editing
the scripts, keeping `PHYS_BATCH * ACCUM_STEPS = 8` to preserve the reproduction's effective batch size:

```bash
PHYS_BATCH=4 ACCUM_STEPS=2 bash run_repro_baseline.sh
```

Caveat: this is not bit-identical to a true single-step BS=8 forward pass — any non-frozen BatchNorm layers in
the backbone compute statistics per physical step (4 images), not per effective batch (8). Close enough for
this purpose, but worth knowing about if results land slightly off from a run that never needed to split.
Deliberately did *not* reach for reducing `--batch_size` outright (changes the reproduction, since BS=8 is one
of Mattia's stated parameters) or mixed-precision training (changes numerics, and Mattia's own script trains in
plain fp32 with no `precision=` argument set — matching that exactly is the point of a reproduction run).
