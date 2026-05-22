#!/usr/bin/env python3
"""
Create one highly compressed NetCDF file from ensemble WRF-SFIRE wrfout files.

Selected static variables are identical across ensemble members, so variables 
such as FXLONG and FXLAT are stored only once, without an ens dimension.

Example:
    python compress_wrf_sfire_ensemble.py \
        --ensemble-dir /path/to/ensemble_runs \
        --member-glob "ens_*" \
        --wrfout-glob "wrfout_d01*" \
        --output wrfsfire_ensemble_subset.nc \
        --vars FXLONG FXLAT FIRE_AREA \
        --static-vars FXLONG FXLAT \
        --float32 \
        --least-significant-digit 4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import xarray as xr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subset and compress ensemble WRF-SFIRE wrfout files into one NetCDF."
    )

    parser.add_argument(
        "--ensemble-dir",
        required=True,
        type=Path,
        help="Directory containing ensemble member folders.",
    )

    parser.add_argument(
        "--member-glob",
        default="*",
        help="Glob pattern for ensemble member folders. Default: '*'.",
    )

    parser.add_argument(
        "--wrfout-glob",
        default="wrfout_d01*",
        help="Glob pattern for wrfout files inside each member folder.",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output compressed NetCDF file.",
    )

    parser.add_argument(
        "--vars",
        nargs="+",
        default=["FXLONG", "FXLAT", "FIRE_AREA"],
        help="Variables to keep from wrfout files.",
    )

    parser.add_argument(
        "--static-vars",
        nargs="+",
        default=["FXLONG", "FXLAT"],
        help=(
            "Static variables that are identical across ensemble members. "
            "These are stored only once, without an ens dimension."
        ),
    )

    parser.add_argument(
        "--chunks-time",
        type=int,
        default=1,
        help="Chunk size along Time. Default: 1.",
    )

    parser.add_argument(
        "--chunks-ens",
        type=int,
        default=1,
        help="Chunk size along ens. Default: 1.",
    )

    parser.add_argument(
        "--compression-level",
        type=int,
        default=9,
        choices=range(0, 10),
        help="NetCDF zlib compression level, 0 to 9. Default: 9.",
    )

    parser.add_argument(
        "--float32",
        action="store_true",
        help="Convert floating-point variables to float32 before writing.",
    )

    parser.add_argument(
        "--least-significant-digit",
        type=int,
        default=None,
        help=(
            "Optional lossy quantization. Example: 4 keeps roughly 4 decimal digits. "
            "This can substantially reduce file size."
        ),
    )

    parser.add_argument(
        "--check-static",
        action="store_true",
        help=(
            "Check that static variables are identical across ensemble members. "
            "This adds I/O cost but is useful for validation."
        ),
    )
    
    parser.add_argument(
        "--srx",
        type=int,
        default=50,
        help=(
            "Number of fire-grid columns to remove from the east/right edge. "
            "Usually srx = fire_dx / atm_dx. Default: 0."
        ),
    )

    parser.add_argument(
        "--sry",
        type=int,
        default=50,
        help=(
            "Number of fire-grid rows to remove from the north/top edge. "
            "Usually sry = fire_dy / atm_dy. Default: 0."
        ),
    )

    return parser.parse_args()


def decode_wrf_times(ds: xr.Dataset) -> xr.Dataset:
    """
    Decode WRF Times character array into a datetime64 Time coordinate.
    """

    if "Times" not in ds:
        return ds

    times = ds["Times"]

    if times.ndim != 2:
        return ds

    try:
        decoded = [
            b"".join(row).decode("utf-8").strip().replace("_", " ")
            for row in times.values.astype("S1")
        ]

        time_values = np.array(decoded, dtype="datetime64[ns]")
        ds = ds.assign_coords(Time=("Time", time_values))
        ds = ds.drop_vars("Times")

    except Exception:
        pass

    return ds


def trim_fire_subgrid_strip(
    ds: xr.Dataset,
    srx: int,
    sry: int,
) -> xr.Dataset:
    """
    Remove the trailing fire-grid strip at the end of the domain.

    srx = fire_dx / atm_dx
    sry = fire_dy / atm_dy

    Assumes WRF-SFIRE fire-grid dimensions:
        west_east_subgrid
        south_north_subgrid
    """

    indexers = {}

    if srx > 0 and "west_east_subgrid" in ds.dims:
        indexers["west_east_subgrid"] = slice(0, -srx)

    if sry > 0 and "south_north_subgrid" in ds.dims:
        indexers["south_north_subgrid"] = slice(0, -sry)

    if indexers:
        ds = ds.isel(indexers)

    return ds


def preprocess_wrfout(
    ds: xr.Dataset,
    variables: Sequence[str],
    static_variables: Sequence[str],
    srx: int = 0,
    sry: int = 0,
) -> xr.Dataset:
    """
    Keep only requested variables and remove Time from static variables.
    """

    keep = [v for v in variables if v in ds.variables]

    if "Times" in ds.variables:
        keep.append("Times")

    ds = ds[keep]
    ds = decode_wrf_times(ds)
    
    ds = trim_fire_subgrid_strip(ds, srx=srx, sry=sry)

    for var in static_variables:
        if var in ds and "Time" in ds[var].dims:
            ds[var] = ds[var].isel(Time=0, drop=True)

    return ds


def open_member_dataset(
    member_dir: Path,
    wrfout_glob: str,
    variables: Sequence[str],
    static_variables: Sequence[str],
    srx: int = 0,
    sry: int = 0,
) -> xr.Dataset:
    """
    Open all wrfout files for one ensemble member.
    """

    files = sorted(member_dir.glob(wrfout_glob))

    if not files:
        raise FileNotFoundError(
            f"No wrfout files found in {member_dir} using pattern {wrfout_glob}"
        )

    ds = xr.open_mfdataset(
        files,
        combine="nested",
        concat_dim="Time",
        preprocess=lambda x: preprocess_wrfout(
            x,
            variables,
            static_variables,
            srx=srx,
            sry=sry,
        ),
        decode_times=False,
        engine="netcdf4",
        parallel=False,
        chunks={},
        data_vars="minimal",
        coords="minimal",
        compat="override",
    )

    if "Time" in ds.coords:
        _, unique_idx = np.unique(ds["Time"].values, return_index=True)
        unique_idx = np.sort(unique_idx)

        if len(unique_idx) != ds.sizes["Time"]:
            ds = ds.isel(Time=unique_idx)

    return ds


def split_static_and_dynamic_vars(
    ds: xr.Dataset,
    static_variables: Sequence[str],
) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Split a member dataset into:
        1. static variables, stored once
        2. dynamic variables, stored per ensemble member
    """

    static_vars_present = [v for v in static_variables if v in ds.data_vars]
    dynamic_vars_present = [v for v in ds.data_vars if v not in static_vars_present]

    ds_static = ds[static_vars_present]
    ds_dynamic = ds[dynamic_vars_present]

    return ds_static, ds_dynamic


def validate_same_static_vars(
    reference_static: xr.Dataset,
    candidate_static: xr.Dataset,
    member_name: str,
) -> None:
    """
    Validate that static variables match the reference member.
    """

    for var in reference_static.data_vars:
        if var not in candidate_static:
            raise ValueError(f"{var} missing from member {member_name}")

        ref = reference_static[var].values
        cand = candidate_static[var].values

        if ref.shape != cand.shape:
            raise ValueError(
                f"{var} shape mismatch for member {member_name}: "
                f"reference {ref.shape}, candidate {cand.shape}"
            )

        if not np.allclose(ref, cand, equal_nan=True):
            raise ValueError(f"{var} differs for member {member_name}")


def maybe_convert_to_float32(ds: xr.Dataset) -> xr.Dataset:
    """
    Convert floating-point variables to float32.
    """

    for var in ds.data_vars:
        if np.issubdtype(ds[var].dtype, np.floating):
            ds[var] = ds[var].astype("float32")

    return ds


def build_encoding(
    ds: xr.Dataset,
    compression_level: int,
    chunks_time: int,
    chunks_ens: int,
    least_significant_digit: int | None,
) -> dict:
    """
    Build NetCDF4 encoding dictionary with aggressive compression.

    Only data variables are compressed. Coordinates are left uncompressed because
    they are small and can be sensitive to stale/incompatible chunk metadata.
    """

    encoding = {}

    for name, da in ds.data_vars.items():
        enc = {}

        # Do not chunk or compress scalar variables.
        if da.ndim == 0:
            encoding[name] = enc
            continue

        enc.update(
            {
                "zlib": True,
                "complevel": compression_level,
                "shuffle": True,
            }
        )

        if np.issubdtype(da.dtype, np.floating):
            enc["_FillValue"] = np.float32(np.nan) if da.dtype == np.float32 else np.nan

            if least_significant_digit is not None:
                enc["least_significant_digit"] = least_significant_digit

        chunksizes = []

        for dim in da.dims:
            dim_size = ds.sizes[dim]

            if dim == "ens":
                chunksizes.append(min(chunks_ens, dim_size))
            elif dim == "Time":
                chunksizes.append(min(chunks_time, dim_size))
            else:
                chunksizes.append(dim_size)

        if len(chunksizes) != da.ndim:
            raise ValueError(
                f"Internal encoding error for variable {name}: "
                f"dims={da.dims}, ndim={da.ndim}, chunksizes={chunksizes}"
            )

        enc["chunksizes"] = tuple(chunksizes)

        encoding[name] = enc

    return encoding


def pad_member_time_dimension(
    dynamic_member_datasets: list[xr.Dataset],
) -> list[xr.Dataset]:
    """
    Pad dynamic member datasets so they all have the same Time length.

    This is a solution for failed WRF-SFIRE ensemble members
    that stopped early and therefore have fewer wrfout time steps.

    It assumes all members use the same output frequency and that member Time
    indices are positionally compatible. Missing trailing times are padded with
    NaN or _FillValue during NetCDF writing.
    """

    max_time_size = max(
        ds.sizes["Time"]
        for ds in dynamic_member_datasets
        if "Time" in ds.dims
    )

    padded = []

    for ds in dynamic_member_datasets:
        if "Time" not in ds.dims:
            padded.append(ds)
            continue

        ntime = ds.sizes["Time"]

        if ntime == max_time_size:
            padded.append(ds)
            continue

        if ntime > max_time_size:
            raise ValueError(
                f"Unexpected Time length {ntime} greater than max_time_size {max_time_size}"
            )

        pad_width = max_time_size - ntime

        print(
            f"Padding member from {ntime} to {max_time_size} Time steps "
            f"with {pad_width} missing trailing steps."
        )

        ds_pad = ds.pad(
            Time=(0, pad_width),
            mode="constant",
            constant_values=np.nan,
        )

        padded.append(ds_pad)

    return padded


def clean_stale_encoding(ds: xr.Dataset) -> xr.Dataset:
    """
    Remove all inherited encoding metadata from all variables and coordinates.

    This is needed after modifying WRF NetCDF variables with operations such as:
        isel(Time=0)
        pad()
        expand_dims()
        concat()
        merge()

    Otherwise xarray/netCDF4 may try to reuse stale chunksizes from the source
    wrfout files, causing:

        ValueError: chunksizes must be a sequence with the same length as dimensions
    """

    ds = ds.copy(deep=False)

    for name in ds.variables:
        ds[name].encoding = {}

    return ds


def main() -> None:
    args = parse_args()

    ensemble_dir = args.ensemble_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()

    member_dirs = sorted(
        p for p in ensemble_dir.glob(args.member_glob)
        if p.is_dir()
    )

    if not member_dirs:
        raise FileNotFoundError(
            f"No ensemble member folders found in {ensemble_dir} "
            f"using pattern {args.member_glob}"
        )

    member_names = [p.name for p in member_dirs]

    reference_static = None
    dynamic_member_datasets = []

    for member_dir in member_dirs:
        print(f"Opening member: {member_dir.name}")

        ds_member = open_member_dataset(
            member_dir=member_dir,
            wrfout_glob=args.wrfout_glob,
            variables=args.vars,
            static_variables=args.static_vars,
            srx=args.srx,
            sry=args.sry,
        )

        ds_static, ds_dynamic = split_static_and_dynamic_vars(
            ds=ds_member,
            static_variables=args.static_vars,
        )

        if reference_static is None:
            reference_static = ds_static

        elif args.check_static:
            validate_same_static_vars(
                reference_static=reference_static,
                candidate_static=ds_static,
                member_name=member_dir.name,
            )

        ds_dynamic = ds_dynamic.expand_dims(ens=[member_dir.name])
        dynamic_member_datasets.append(ds_dynamic)

    if reference_static is None:
        raise RuntimeError("No reference static dataset was created.")

    print("Padding incomplete ensemble members along Time...")

    dynamic_member_datasets = pad_member_time_dimension(dynamic_member_datasets)

    print("Concatenating dynamic ensemble variables...")

    ds_dynamic_all = xr.concat(
        dynamic_member_datasets,
        dim="ens",
        data_vars="all",
        coords="minimal",
        compat="override",
        join="override",
    )

    ds_dynamic_all = ds_dynamic_all.assign_coords(ens=("ens", member_names))

    print("Merging static variables and dynamic ensemble variables...")

    ds_out = xr.merge(
        [reference_static, ds_dynamic_all],
        compat="override",
        join="outer",
    )

    if args.float32:
        ds_out = maybe_convert_to_float32(ds_out)
        
    if "Times" in ds_out:
        print("Dropping Times variable before writing to avoid string chunking issues.")
        ds_out = ds_out.drop_vars("Times")
        
    ds_out = clean_stale_encoding(ds_out)
    
    ds_out.attrs.update(
        {
            "title": "Subsetted compressed WRF-SFIRE ensemble output",
            "source": "WRF-SFIRE wrfout files",
            "subset_variables": ", ".join(args.vars),
            "static_variables_stored_once": ", ".join(args.static_vars),
            "ensemble_members": ", ".join(member_names),
        }
    )

    encoding = build_encoding(
        ds=ds_out,
        compression_level=args.compression_level,
        chunks_time=args.chunks_time,
        chunks_ens=args.chunks_ens,
        least_significant_digit=args.least_significant_digit,
    )

    print("Output variables and dimensions:")
    for name in ds_out.variables:
        print(f"  {name}: dims={ds_out[name].dims}, shape={ds_out[name].shape}")

    print("Explicit NetCDF encodings:")
    for name, enc in encoding.items():
        print(f"  {name}: dims={ds_out[name].dims}, chunksizes={enc.get('chunksizes')}")

    print(f"Writing compressed NetCDF: {output}")

    ds_out.to_netcdf(
        output,
        engine="netcdf4",
        format="NETCDF4",
        encoding=encoding,
    )

    print("Done.")


if __name__ == "__main__":
    main()