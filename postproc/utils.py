from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from glob import glob
import os.path as osp
import xarray as xr
import numpy as np
import subprocess
import re


def is_sim_success(log_path):
    # Verify if the simulation was successful
    last_line = open(log_path).readlines()[-1]
    if "SUCCESS COMPLETE WRF" in last_line:
        return True
    return False 


def find_perturbations(log_path, perturb_vars):
    # Search rsl.out.0000 for the line containing 'stoch perturbation {var}'
    # and return the numeric value, or None if not found.
    perturbs = {}
    if not osp.exists(log_path):
        return None
    for var in perturb_vars:
        base_pattern = f"pert. {var}"
        result = subprocess.run(
            ["grep", base_pattern, log_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or not result.stdout:
            perturbs.update({var: None})
        pattern = re.compile(
            rf"{re.escape(base_pattern)}[^:]*:\s*"
            r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            re.IGNORECASE,
        )
        for line in result.stdout.splitlines():
            m = pattern.search(line)
            if m:
                perturbs.update({var: float(m.group(1))})
    return perturbs


def process_ensemble(ens_base_path, stoch_vars):
    ens_paths = sorted(glob(osp.join(ens_base_path, "*")))
    data = {}
    for k, ens_path in enumerate(ens_paths):
        ens_id = osp.basename(ens_path)
        print(f"Processing ensemble {ens_id}")
        log_file = osp.join(ens_path, "rsl.out.0000")
        perturbs = find_perturbations(log_file, stoch_vars)
        if is_sim_success(log_file):
            files = sorted(glob(osp.join(ens_path, "wrfout_d03*")))
            last_file = files[-1]
            ds = xr.open_dataset(last_file)
            if k == 0:
                srx = int(ds.dims["west_east_subgrid"]/(ds.dims["west_east"]+1))
                sry = int(ds.dims["south_north_subgrid"]/(ds.dims["south_north"]+1))
                lons = ds.variables["FXLONG"][-1, :-sry, :-srx]
                lats = ds.variables["FXLAT"][-1, :-sry, :-srx]
                data.update({"LONS": lons, "LATS": lats, "MEMBERS": {}, "WRONG_MEMBERS": {}})
            tign_g = ds.variables["TIGN_G"][-1, :-sry, :-srx].data
            fire_area = ds.variables["FIRE_AREA"][-1, :-sry, :-srx]
            data["MEMBERS"].update({
                    ens_id: { 
                        "TIGN_G": tign_g, "FIRE_AREA": fire_area,
                        "PERTURBATIONS": perturbs
                    }
            })
        else:
            print(f"WARNING: ensemble {ens_id} did not run to completion!")
            if "WRONG_MEMBERS" not in data.keys():
                data.update({"WRONG_MEMBERS": {}})
            data["WRONG_MEMBERS"].update({
                    ens_id: { 
                        "PERTURBATIONS": perturbs
                    }
            }) 
    if "MEMBERS" in data.keys():
        n_members = len(data["MEMBERS"])
        fire_prob = np.sum([v["FIRE_AREA"].data for v in data["MEMBERS"].values()], axis=0) / n_members
        data["FIRE_PROB"] = fire_prob
        ens_mean = np.mean([v["TIGN_G"].data for v in data["MEMBERS"].values()], axis=0)
        data["ENS_MEAN_TIGN"] = ens_mean
        ens_std = np.std([v["TIGN_G"].data for v in data["MEMBERS"].values()], axis=0)
        data["ENS_STD_TIGN"] = ens_std
    return data


def plot_baseline_vs_control(orig_path, ctrl_path):
    orig_ds = xr.open_dataset(orig_path)
    ctrl_ds = xr.open_dataset(ctrl_path)
    srx = int(ctrl_ds.dims["west_east_subgrid"]/(ctrl_ds.dims["west_east"]+1))
    sry = int(ctrl_ds.dims["south_north_subgrid"]/(ctrl_ds.dims["south_north"]+1))
    lons = ctrl_ds.variables["FXLONG"][-1, :-sry, :-srx]
    lats = ctrl_ds.variables["FXLAT"][-1, :-sry, :-srx]
    nfuel_cat = ctrl_ds.variables["NFUEL_CAT"][-1, :-sry, :-srx]

    ctrl_fa = ctrl_ds.variables["FIRE_AREA"][-1, :-sry, :-srx]
    orig_fa = orig_ds.variables["FIRE_AREA"][-1, :-sry, :-srx]
    fig, ax = plt.subplots(figsize=(10, 6))
    cfax = ax.contourf(
        lons, lats, nfuel_cat, 
        cmap="tab20", shading="nearest"
    )
    plt.colorbar(cfax, label="Fuel Category")
    pc_orig = ax.contour(
        lons, lats, orig_fa, 
        levels = [0.5], colors="blue",
        shading="nearest", 
        antialiased=False
    )
    pc_ctrl = ax.contour(
        lons, lats, ctrl_fa, 
        levels = [0.5], colors="black", 
        shading="nearest",
        antialiased=False
    )
    custom_lines = [
        Line2D([0], [0], color="blue"), Line2D([0], [0], color="black")
    ]
    plt.legend(custom_lines, ["SIM_BASELINE", "SIM_CONTROL"])
    plt.grid(False)


def plot_perturbations(ensemble):
    members = list(ensemble.keys())
    members_id = range(len(members))
    perturbs = {}
    for ens in ensemble.values(): 
        for k,v in ens["PERTURBATIONS"].items():
            if k not in perturbs:
                perturbs.update({k: [v]})
            else:
                perturbs[k].append(v)
                
    cmap = plt.cm.RdBu_r
    for k, perturb in perturbs.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        v_abs = np.max(np.abs(perturb))  # symmetric range around 0
        norm = TwoSlopeNorm(vmin=-v_abs, vcenter=0.0, vmax=v_abs)
        for m, p in zip(members_id, perturb):
            color = cmap(norm(p))
            ax.plot([m, m], [0, p], 
                color=color, linewidth=2)
            ax.scatter(m, p, s=50, 
                facecolor=color,
                edgecolors="k",
                linewidths=0.8,
                zorder=3)
            
        ax.axhline(0, color="k", linewidth=1)  # zero line
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(
            sm,
            ax=ax,
            orientation="horizontal",
            pad=0.15,
            aspect=40,
        )
        cb.set_label("negative <-  perturbation  -> positive")
        cb.set_ticks(np.linspace(-v_abs, v_abs, 5))
        ax.set_xticks(members_id)
        ax.set_xticklabels(members, rotation=45)
        ax.set_ylabel("{} perturbation".format(k))
        plt.tight_layout()
        plt.show()
        
        
def plot_fire_isochrones(data, xlim=[], ylim=[]):
    fig, ax = plt.subplots(figsize=(10, 8))
    for k,v in data["MEMBERS"].items():
        ax.contour(data["LONS"], data["LATS"], v["FIRE_AREA"], levels=[0.1], colors='gray')
    ax.contour(data["LONS"], data["LATS"], data["FIRE_PROB"], levels=[0.5], colors='r')
    ax.grid(False)
    if len(xlim):
        ax.set_xlim(xlim)
    if len(ylim):
        ax.set_ylim(ylim)
    custom_lines = [
        Line2D([0], [0], color='gray'), Line2D([0], [0], color='r'), 
        Line2D([0], [0], color='k', linestyle='--')
    ]
    plt.legend(custom_lines, ["Ensemble Realizations", "Ensemble Mean"])
    plt.xlabel("Longitudes")
    plt.ylabel("Latitudes")
    plt.show()
    

def plot_fire_prob(data, xlim=[], ylim=[]):
    fig, ax = plt.subplots(figsize=(10, 8))
    # Diverging colormap with white in the middle
    cmap = plt.cm.RdBu_r               # has white-ish center
    norm = TwoSlopeNorm(vmin=0, vcenter=50, vmax=100)
    levels = np.arange(0, 101, 10)
    pc = ax.contourf(
        data["LONS"], data["LATS"],
        100 * data["FIRE_PROB"],
        levels=levels,
        cmap=cmap, norm=norm,
        extend="both",
        alpha=0.8, shading="nearest", 
        edgecolors="none", 
        antialiased=False
    )
    # Horizontal colorbar with arrows in the label
    cb = fig.colorbar(
        pc,
        ax=ax,
        orientation="horizontal",
        pad=0.12,                      # distance from map
        aspect=40,                     # long & thin
    )
    cb.set_label("faster <-  fire probability  -> slower")
    cb.set_ticks([0, 25, 50, 75, 100])   # or np.arange(0, 101, 10)
    if len(xlim):
        ax.set_xlim(xlim)
    if len(ylim):
        ax.set_ylim(ylim)
    ax.grid(False)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    pc = ax.contourf(
        data["LONS"], data["LATS"], 100 * data["FIRE_PROB"], 
        levels = np.arange(0, 101, 10),
        cmap="jet", alpha=0.6, shading="nearest", 
        edgecolors="none", antialiased=False
    )
    pc.set_rasterized(True)
    cb = plt.colorbar(pc)
    cb.set_label("Fire Probability")
    if len(xlim):
        ax.set_xlim(xlim)
    if len(ylim):
        ax.set_ylim(ylim)
    plt.grid(False)