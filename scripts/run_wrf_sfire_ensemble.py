#!/usr/bin/env python3
"""
Prepare and launch a WRF ensemble on SLURM.

Steps per ensemble member:
1) Create a template WRF run directory.
2) Modify stoch parameters in the namelist.input.
3) Submit the job via 'sbatch sim-script.sub'.

Usage example:
    python run_wrf_ensemble.py \
        --template-run-dir /path/to/template_run \
        --ensemble-root /path/to/ensemble_runs \
        --n-members 20
"""

from pathlib import Path
import numpy as np
import subprocess
import argparse
import shutil
import f90nml
import sys
import os

# Template for the SLURM script
SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=WRF-SFIRE
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks={total_ntasks}

# load appropriate modules
source ~/.wrf

{mpirun_block}
"""

base_seed = 12345
parameters = {
    "fgi": {
        "min": -50.0,
        "max": 50.0,
        "std": 30.0,
    },
    "fmc": {
        "min": -50.0,
        "max": 50.0,
        "std": 30.0,
    },
    "u_wind": {
        "min": -2.0,
        "max": 2.0,
        "std": 0.5,
    },
    "v_wind": {
        "min": -2.0,
        "max": 2.0,
        "std": 0.5,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Setup and submit WRF ensemble runs.")
    parser.add_argument(
        "--template-run-dir",
        required=True,
        type=Path,
        help="Directory containing the template WRF run (namelist.input, real.exe, etc.).",
    )
    parser.add_argument(
        "--ensemble-root",
        required=True,
        type=Path,
        help="Directory where ensemble member folders will be created.",
    )
    parser.add_argument(
        "--n-members",
        type=int,
        required=True,
        help="Number of ensemble members (will create members 0..N-1).",
    )
    parser.add_argument(
        "--cores-per-run", 
        type=int, 
        required=True,
        help="MPI ranks per WRF run (mpirun -np)"
    )
    parser.add_argument(
        "--runs-per-job", 
        type=int, 
        required=True,
        help="Number of WRF runs to pack into a single Slurm job"
    )
    parser.add_argument(
        "--params",
        type=str,
        default=",".join(parameters.keys()),
        help="Parameters to perturb, available: fgi, fmc, u_wind and/or v_wind.",
    )
    parser.add_argument(
        "--member-prefix",
        type=str,
        default="ens",
        help="Prefix for member directories (e.g., 'ens' -> ens_000, ens_001...).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions but do not actually run real.exe / preproc / sbatch.",
    )
    return parser.parse_args()


def copy_template(template_dir: Path, dest_dir: Path):
    if dest_dir.exists():
        raise FileExistsError(f"Destination directory already exists: {dest_dir}")
    shutil.copytree(template_dir, dest_dir)
    

def draw_random_value(min_value, max_value, std, rng=None, mean=None, max_attempts=10_000):
    """
    Draw one random value from a normal distribution bounded by min_value and max_value.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    if mean is None:
        mean = 0.5 * (min_value + max_value)
        
    for _ in range(max_attempts):
        value = rng.normal(loc=mean, scale=std)
        if min_value <= value <= max_value:
            return value

    raise RuntimeError(
        "Could not draw a value inside the requested bounds. "
        "Check min_value, max_value, std, and mean."
    )    
    

def set_random_stoch_params(ensemble_member, namelist_path, params):
    rng = np.random.default_rng(base_seed + ensemble_member)
    random_values = {}
    for name, config in parameters.items():
        random_values[name] = draw_random_value(
            min_value=config["min"],
            max_value=config["max"],
            std=config["std"],
            rng=rng,
        )
    nml = f90nml.read(namelist_path)
    nml["fire"]["use_stoch_inputs"] = 1
    if "fgi" in params:
        nml["fire"]["stoch_fuel_load_value"] = random_values["fgi"]
    if "fmc" in params:
        nml["fire"]["stoch_fmc_value"] = random_values["fmc"]
    if "u_wind" in params:
        nml["fire"]["stoch_wind_value_u"] = random_values["u_wind"]
    if "v_wind" in params:
        nml["fire"]["stoch_wind_value_v"] = random_values["v_wind"]
    f90nml.write(nml, namelist_path, force=True) 
    return random_values    


def run_real(member_dir: Path, dry_run: bool = False):
    exe = member_dir / "real.exe"
    if not exe.exists():
        raise FileNotFoundError(f"real.exe not found in {member_dir}")
    if dry_run:
        print(f"[DRY-RUN] Would run: ./real.exe (cwd={member_dir})")
        return
    print(f"Running real.exe in {member_dir} ...")
    subprocess.run(["./real.exe"], cwd=member_dir, check=True)


def run_preproc(member_dir: Path, preproc_script: Path, dry_run: bool = False):
    if not preproc_script.exists():
        raise FileNotFoundError(f"Preprocessing script not found: {preproc_script}")
    if dry_run:
        print(f"[DRY-RUN] Would run: python {preproc_script.name} (cwd={member_dir})")
        return
    # Copy or symlink the script into the member directory? Usually you can just call it via full path.
    print(f"Running preprocessing script in {member_dir} ...")
    subprocess.run(
        [sys.executable, str(preproc_script)],
        cwd=member_dir,
        check=True,
    )
    

def submit_batch(run_dirs, cores_per_run, slurm_dir, dry_run=False):
    """
    run_dirs: list of run directories to execute in one Slurm job
    cores_per_run: MPI ranks per WRF run (mpirun -np)
    slurm_dir: where to write the Slurm script (can be any directory you like)
    dry_run: generate everything but not submit
    """
    total_ntasks = cores_per_run * len(run_dirs)

    # Build a block of mpirun commands, one per run_dir, all in background
    mpirun_lines = []
    for d in run_dirs:
        abs_d = os.path.abspath(d)
        mpirun_lines.append(
            f"(cd {abs_d} && mpirun -np {cores_per_run} ./wrf.exe) &"
        )
    mpirun_lines.append("wait")
    mpirun_block = "\n".join(mpirun_lines)
    slurm_path = os.path.join(slurm_dir, "sim-script.sub")
    with open(slurm_path, "w") as f:
        f.write(SLURM_TEMPLATE.format(
            total_ntasks=total_ntasks,
            mpirun_block=mpirun_block,
        ))
    if dry_run:
        print(f"[DRY-RUN] Would run: sbatch sim-script.sub (cwd={slurm_dir})")
        return
    print(f"Submitting SLURM job in {slurm_dir} ...")
    subprocess.run(["sbatch", slurm_path], cwd=slurm_dir, check=True)
    

def main():
    args = parse_args()

    template_dir = args.template_run_dir.resolve()
    ensemble_root = args.ensemble_root.resolve()
    cores_per_run = args.cores_per_run
    runs_per_job = args.runs_per_job
    params = args.params

    if not template_dir.is_dir():
        raise NotADirectoryError(f"Template run dir does not exist: {template_dir}")

    ensemble_root.mkdir(parents=True, exist_ok=True)
    params = params.split(",")

    print(f"Template directory: {template_dir}")
    print(f"Ensemble root: {ensemble_root}")
    print(f"Members: 0 .. {args.n_members - 1}")
    print(f"Cores per member: {cores_per_run}")
    print(f"Runs per job: {runs_per_job}")
    print(f"Parameters to perturb: {params}")
    if args.dry_run:
        print("DRY-RUN MODE: No commands will actually be executed.\n")

    # [1] Copy all the content and prepare stochastic runs
    member_dirs = []
    for member_id in range(args.n_members):
        member_name = f"{args.member_prefix}_{member_id:03d}"
        member_dir = ensemble_root / member_name

        print(f"\n=== Member {member_id} -> {member_name} ===")

        # 1) Copy template
        print(f"Copying template to {member_dir} ...")
        copy_template(template_dir, member_dir)

        # 2) Create stochastic options for namelist.input
        namelist_path = member_dir / "namelist.input"
        random_params = set_random_stoch_params(
            member_id, namelist_path, params
        )
        print(f"Set params = {random_params} in {namelist_path}")        
        member_dirs.append(member_dir)

    # [2] Submit all the members in groups depending on input options
    if runs_per_job == 1:
        # one member per job
        for d in member_dirs:
            try:
                submit_batch([d], cores_per_run, 
                            slurm_dir=d, dry_run=args.dry_run)
            except subprocess.CalledProcessError as e:
                print(f"ERROR: sbatch failed for member {member_id}: {e}")
                # Continue to next member
                continue 
    else:
        # multiple members per job
        for i in range(0, len(member_dirs), runs_per_job):
            batch = member_dirs[i : i + runs_per_job]
            # You can choose any directory to hold the batch slurm file.
            # Using the first member's directory is simple and works.
            slurm_dir = batch[0]
            try:
                submit_batch(batch, cores_per_run, 
                            slurm_dir=slurm_dir, 
                            dry_run=args.dry_run)
            except subprocess.CalledProcessError as e:
                print(f"ERROR: sbatch failed for member {member_id}: {e}")
                # Continue to next member
                continue
                
    print("\nDone setting up ensemble.")

if __name__ == "__main__":
    main()
