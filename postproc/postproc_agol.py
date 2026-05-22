#!/usr/bin/env python3
"""
Generate ArcGIS-upload-ready vector files from a compressed WRF-SFIRE ensemble NetCDF.

Inputs expected:
    FIRE_AREA(ens, Time, south_north_subgrid, west_east_subgrid)
    FXLONG(south_north_subgrid, west_east_subgrid)
    FXLAT(south_north_subgrid, west_east_subgrid)

Outputs:
    fire_probability.geojson
    hourly_fire_progression.geojson
    wrfsfire_arcgis_upload.gpkg

Optional:
    zipped Shapefiles for manual ArcGIS upload.

Example:
    python prepare_arcgis_fire_layers.py \
        --input forest_wrfsfire_ens.nc \
        --output-dir arcgis_upload \
        --threshold 0 \
        --progression-mode cumulative \
        --progression-prob-threshold 0.5 \
        --write-shapefiles
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import shape
from shapely.ops import unary_union
from rasterio import features
from rasterio.transform import from_bounds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare WRF-SFIRE ensemble fire layers for manual ArcGIS upload."
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Compressed WRF-SFIRE ensemble NetCDF file.",
    )

    parser.add_argument(
        "--output-dir",
        default=Path("arcgis_upload"),
        type=Path,
        help="Output directory for ArcGIS upload files.",
    )

    parser.add_argument(
        "--fire-var",
        default="FIRE_AREA",
        help="Name of fire variable. Default: FIRE_AREA.",
    )

    parser.add_argument(
        "--lon-var",
        default="FXLONG",
        help="Name of longitude variable. Default: FXLONG.",
    )

    parser.add_argument(
        "--lat-var",
        default="FXLAT",
        help="Name of latitude variable. Default: FXLAT.",
    )

    parser.add_argument(
        "--threshold",
        default=0.0,
        type=float,
        help="Cell is treated as burned when FIRE_AREA > threshold. Default: 0.",
    )

    parser.add_argument(
        "--probability-mode",
        choices=["any_time", "final_time"],
        default="any_time",
        help=(
            "Fire probability definition. "
            "'any_time' means probability of burning at least once during the simulation. "
            "'final_time' means probability of being burned at the final available time."
        ),
    )

    parser.add_argument(
        "--probability-bins",
        nargs="+",
        type=float,
        default=[0.01, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.01],
        help=(
            "Probability bin edges. Default creates 10 classes: "
            "1-10, 10-20, ..., 90-100 percent. "
            "The first bin starts at 0.01 to exclude zero-probability areas."
        ),
    )

    parser.add_argument(
        "--progression-mode",
        choices=["cumulative", "new_growth"],
        default="cumulative",
        help=(
            "Hourly progression polygon mode. "
            "'cumulative' writes burned extent at each hour. "
            "'new_growth' writes only newly burned area at each hour."
        ),
    )

    parser.add_argument(
        "--progression-prob-threshold",
        default=0.5,
        type=float,
        help=(
            "Ensemble probability threshold used for hourly progression. "
            "Default: 0.5 means cells burned in at least 50 percent of available members."
        ),
    )

    parser.add_argument(
        "--area-crs",
        default="EPSG:5070",
        help=(
            "Projected CRS used to calculate polygon area. "
            "Default EPSG:5070, NAD83 / Conus Albers."
        ),
    )

    parser.add_argument(
        "--write-shapefiles",
        action="store_true",
        help="Also write zipped Shapefiles for manual ArcGIS upload.",
    )

    parser.add_argument(
        "--skip-geojson",
        action="store_true",
        help="Skip GeoJSON output.",
    )

    parser.add_argument(
        "--skip-gpkg",
        action="store_true",
        help="Skip GeoPackage output.",
    )

    parser.add_argument(
        "--min-area-ha",
        default=0.0,
        type=float,
        help="Drop polygons smaller than this area in hectares. Default: 0.",
    )

    return parser.parse_args()


def add_probability_style_fields(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Add 10-class style/color fields to fire probability polygons.

    Colors follow a blue, light-blue, near-white, orange, red progression.
    """

    gdf = gdf.copy()

    color_map = {
        "1-10%": "#08306b",
        "10-20%": "#2171b5",
        "20-30%": "#6baed6",
        "30-40%": "#bdd7e7",
        "40-50%": "#f7f7f7",
        "50-60%": "#fee8c8",
        "60-70%": "#fdbb84",
        "70-80%": "#fc8d59",
        "80-90%": "#e34a33",
        "90-100%": "#b30000",
    }

    gdf["fill_col"] = gdf["pclass"].map(color_map).fillna("#808080")
    gdf["line_col"] = "#000000"
    gdf["opacity"] = 0.75
    gdf["line_w"] = 0.2

    return gdf


def add_progression_style_fields(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Add style/color fields to hourly fire progression polygons.
    """

    gdf = gdf.copy()

    max_hour = max(float(gdf["hour"].max()), 1.0)

    def hour_to_color(hour):
        # Simple yellow-orange-red ramp encoded manually.
        x = float(hour) / max_hour

        if x < 0.33:
            return "#fee08b"
        elif x < 0.66:
            return "#f46d43"
        else:
            return "#a50026"

    gdf["fill"] = gdf["hour"].apply(hour_to_color)
    gdf["stroke"] = "#000000"
    gdf["fill_opacity"] = 0.70
    gdf["stroke_opacity"] = 0.20
    gdf["stroke_width"] = 0.2

    return gdf


def build_approx_transform(lon2d: np.ndarray, lat2d: np.ndarray):
    """
    Build an approximate affine transform from 2D lon/lat bounds.

    This assumes the fire grid is approximately rectilinear in lon/lat. For many
    small WRF-SFIRE domains this is acceptable for visualization/upload purposes.
    If your domain is very large or strongly curvilinear, consider regridding
    the masks to a regular projected raster before polygonization.
    """

    west = float(np.nanmin(lon2d))
    east = float(np.nanmax(lon2d))
    south = float(np.nanmin(lat2d))
    north = float(np.nanmax(lat2d))

    height, width = lon2d.shape

    return from_bounds(
        west=west,
        south=south,
        east=east,
        north=north,
        width=width,
        height=height,
    )


def needs_vertical_flip(lat2d: np.ndarray) -> bool:
    return float(np.nanmean(lat2d[-1, :])) > float(np.nanmean(lat2d[0, :]))


def polygonize_boolean_mask(
    mask: np.ndarray,
    transform,
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """
    Convert a 2D boolean mask to dissolved polygons.
    """

    mask = np.asarray(mask, dtype=bool)

    # Minimal WRF/SFIRE orientation fix:
    # rasterio assumes row 0 is north/top, while south_north arrays often
    # have row 0 at the south/bottom side.
    mask = np.flipud(mask)

    if not mask.any():
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)

    mask_uint8 = mask.astype("uint8")

    geometries = []

    for geom, value in features.shapes(
        mask_uint8,
        mask=mask,
        transform=transform,
    ):
        if value == 1:
            geometries.append(shape(geom))

    if not geometries:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)

    dissolved = unary_union(geometries)

    if dissolved.is_empty:
        final_geometries = []
    elif dissolved.geom_type == "Polygon":
        final_geometries = [dissolved]
    elif dissolved.geom_type == "MultiPolygon":
        final_geometries = list(dissolved.geoms)
    else:
        final_geometries = []

    return gpd.GeoDataFrame({"geometry": final_geometries}, geometry="geometry", crs=crs)


def add_area_fields(
    gdf: gpd.GeoDataFrame,
    area_crs: str,
) -> gpd.GeoDataFrame:
    """
    Add area_m2 and area_ha fields using a projected CRS.
    """

    if len(gdf) == 0:
        gdf["area_m2"] = []
        gdf["area_ha"] = []
        return gdf

    gdf = gdf.copy()
    area = gdf.to_crs(area_crs).area

    gdf["area_m2"] = area.astype(float)
    gdf["area_ha"] = (area / 10000.0).astype(float)

    return gdf


def drop_small_polygons(
    gdf: gpd.GeoDataFrame,
    min_area_ha: float,
) -> gpd.GeoDataFrame:
    """
    Drop polygons smaller than min_area_ha.
    """

    if min_area_ha <= 0 or len(gdf) == 0:
        return gdf

    if "area_ha" not in gdf:
        raise ValueError("area_ha field is required before applying min_area_ha filter.")

    return gdf.loc[gdf["area_ha"] >= min_area_ha].copy()


def get_time_strings(ds: xr.Dataset) -> list[str]:
    """
    Return time values as ArcGIS-friendly ISO strings.
    """

    ntime = ds.sizes["Time"]

    if "Time" in ds.coords:
        raw = ds["Time"].values
        times = pd.to_datetime(raw, errors="coerce")

        if not pd.isna(times).all():
            return [str(t) for t in times]

    if "XTIME" in ds:
        raw = ds["XTIME"].values
        return [str(v) for v in raw]

    return [f"hour_{i:03d}" for i in range(ntime)]


def compute_probability_any_time(
    fire: xr.DataArray,
    threshold: float,
) -> xr.DataArray:
    """
    Probability that a cell burns at least once during the simulation.

    NaN padded values do not burn.
    """

    burned = fire > threshold
    burned_any = burned.any(dim="Time")
    prob = burned_any.mean(dim="ens", skipna=True)

    return prob


def compute_probability_final_time(
    fire: xr.DataArray,
    threshold: float,
) -> xr.DataArray:
    """
    Probability that a cell is burned at the final Time index.

    NaN padded members are excluded from the denominator.
    """

    final_fire = fire.isel(Time=-1)
    valid = np.isfinite(final_fire)
    burned = final_fire > threshold

    burned_count = burned.where(valid).sum(dim="ens", skipna=True)
    valid_count = valid.sum(dim="ens")

    prob = burned_count / valid_count.where(valid_count > 0)

    return prob


def build_fire_probability_layer(
    ds: xr.Dataset,
    fire_var: str,
    lon_var: str,
    lat_var: str,
    threshold: float,
    probability_mode: str,
    probability_bins: list[float],
    area_crs: str,
    min_area_ha: float,
) -> gpd.GeoDataFrame:
    """
    Build dissolved probability class polygons.
    """

    fire = ds[fire_var]
    lon = ds[lon_var].values
    lat = ds[lat_var].values
    transform = build_approx_transform(lon, lat)
    flip_y = needs_vertical_flip(lat)

    print(f"Computing fire probability using mode: {probability_mode}")

    if probability_mode == "any_time":
        prob = compute_probability_any_time(fire, threshold)
    elif probability_mode == "final_time":
        prob = compute_probability_final_time(fire, threshold)
    else:
        raise ValueError(f"Unsupported probability_mode: {probability_mode}")

    prob_values = prob.compute().values

    layers = []
    bins = probability_bins

    if len(bins) < 2:
        raise ValueError("At least two probability bin edges are required.")

    for pmin, pmax in zip(bins[:-1], bins[1:]):
        upper = min(pmax, 1.0)

        if pmax > 1.0:
            mask = (prob_values >= pmin) & (prob_values <= 1.0)
        else:
            mask = (prob_values >= pmin) & (prob_values < pmax)

        if not np.any(mask):
            continue

        label = f"{int(round(pmin * 100))}-{int(round(upper * 100))}%"

        print(f"Polygonizing probability class {label}")

        gdf_bin = polygonize_boolean_mask(mask, transform=transform)

        if len(gdf_bin) == 0:
            continue

        gdf_bin["pmin"] = float(pmin)
        gdf_bin["pmax"] = float(upper)
        gdf_bin["pclass"] = label
        gdf_bin["thresh"] = float(threshold)
        gdf_bin["pmode"] = probability_mode
        gdf_bin["n_ens"] = int(ds.sizes["ens"])

        gdf_bin = add_area_fields(gdf_bin, area_crs=area_crs)
        gdf_bin = drop_small_polygons(gdf_bin, min_area_ha=min_area_ha)

        if len(gdf_bin) > 0:
            layers.append(gdf_bin)

    if not layers:
        return gpd.GeoDataFrame(
            columns=[
                "pmin",
                "pmax",
                "pclass",
                "thresh",
                "pmode",
                "n_ens",
                "area_m2",
                "area_ha",
                "geometry",
            ],
            geometry="geometry",
            crs="EPSG:4326",
        )

    gdf = pd.concat(layers, ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")

    return gdf


def compute_hourly_probability(
    fire_t: xr.DataArray,
    threshold: float,
) -> xr.DataArray:
    """
    Compute hourly burn probability over ensemble members.

    NaN padded failed members are excluded from the denominator for that hour.
    """

    valid = np.isfinite(fire_t)
    burned = fire_t > threshold

    burned_count = burned.where(valid).sum(dim="ens", skipna=True)
    valid_count = valid.sum(dim="ens")

    prob = burned_count / valid_count.where(valid_count > 0)

    return prob


def build_hourly_progression_layer(
    ds: xr.Dataset,
    fire_var: str,
    lon_var: str,
    lat_var: str,
    threshold: float,
    progression_mode: str,
    progression_prob_threshold: float,
    area_crs: str,
    min_area_ha: float,
) -> gpd.GeoDataFrame:
    """
    Build hourly fire progression polygons.

    progression_mode:
        cumulative:
            Polygonize cells whose ensemble burn probability is above the
            threshold at each hour.

        new_growth:
            Polygonize only the newly burned area relative to the previous hour.
    """

    fire = ds[fire_var]
    lon = ds[lon_var].values
    lat = ds[lat_var].values
    transform = build_approx_transform(lon, lat)


    time_strings = get_time_strings(ds)

    layers = []
    previous_extent = None

    for i in range(0, ds.sizes["Time"], 4):
        print(f"Processing hourly progression step {i + 1}/{ds.sizes['Time']}")

        fire_t = fire.isel(Time=i)

        prob_t = compute_hourly_probability(
            fire_t=fire_t,
            threshold=threshold,
        )

        extent = (prob_t.compute().values >= progression_prob_threshold)
        extent = np.asarray(extent, dtype=bool)

        if progression_mode == "cumulative":
            mask = extent

        elif progression_mode == "new_growth":
            if previous_extent is None:
                mask = extent
            else:
                mask = extent & ~previous_extent

        else:
            raise ValueError(f"Unsupported progression_mode: {progression_mode}")

        previous_extent = extent.copy()

        if not np.any(mask):
            continue

        print(f"Polygonizing hour {i}")

        gdf_i = polygonize_boolean_mask(mask, transform=transform)

        if len(gdf_i) == 0:
            continue

        gdf_i["hour"] = int((i - 1) // 4)
        gdf_i["vtime"] = time_strings[i]
        gdf_i["thresh"] = float(threshold)
        gdf_i["pthresh"] = float(progression_prob_threshold)
        gdf_i["mode"] = progression_mode

        gdf_i = add_area_fields(gdf_i, area_crs=area_crs)
        gdf_i = drop_small_polygons(gdf_i, min_area_ha=min_area_ha)

        if len(gdf_i) > 0:
            layers.append(gdf_i)

    if not layers:
        return gpd.GeoDataFrame(
            columns=[
                "hour",
                "vtime",
                "thresh",
                "pthresh",
                "mode",
                "area_m2",
                "area_ha",
                "geometry",
            ],
            geometry="geometry",
            crs="EPSG:4326",
        )

    gdf = pd.concat(layers, ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")

    return gdf


def clean_for_arcgis(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Clean field types and geometries for manual ArcGIS upload.
    """

    gdf = gdf.copy()

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    # Remove empty or null geometries.
    gdf = gdf.loc[gdf.geometry.notnull()].copy()
    gdf = gdf.loc[~gdf.geometry.is_empty].copy()

    # Fix invalid geometries using a zero-width buffer fallback.
    if len(gdf) > 0:
        invalid = ~gdf.geometry.is_valid
        if invalid.any():
            gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)

    # Convert unsupported field types.
    for col in gdf.columns:
        if col == gdf.geometry.name:
            continue

        if pd.api.types.is_datetime64_any_dtype(gdf[col]):
            gdf[col] = pd.to_datetime(gdf[col]).astype(str)

        elif pd.api.types.is_bool_dtype(gdf[col]):
            gdf[col] = gdf[col].astype("int16")

        elif pd.api.types.is_object_dtype(gdf[col]):
            gdf[col] = gdf[col].astype(str)

    return gdf


def write_zipped_shapefile(
    gdf: gpd.GeoDataFrame,
    output_zip: Path,
    layer_name: str,
) -> None:
    """
    Write a zipped Shapefile suitable for ArcGIS manual upload.
    """

    output_zip = Path(output_zip)
    work_dir = output_zip.with_suffix("")

    if work_dir.exists():
        shutil.rmtree(work_dir)

    work_dir.mkdir(parents=True, exist_ok=True)

    shp_path = work_dir / f"{layer_name}.shp"

    gdf.to_file(shp_path, driver="ESRI Shapefile")

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in work_dir.glob(f"{layer_name}.*"):
            zf.write(path, arcname=path.name)

    print(f"Wrote {output_zip}")


def write_outputs(
    gdf_prob: gpd.GeoDataFrame,
    gdf_prog: gpd.GeoDataFrame,
    output_dir: Path,
    write_geojson: bool,
    write_gpkg: bool,
    write_shapefiles: bool,
) -> None:
    """
    Write all requested GIS upload files.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    gdf_prob = clean_for_arcgis(gdf_prob)
    gdf_prog = clean_for_arcgis(gdf_prog)
    
    gdf_prob = add_probability_style_fields(gdf_prob)
    gdf_prog = add_progression_style_fields(gdf_prog)
    
    if write_geojson:
        prob_geojson = output_dir / "fire_probability.geojson"
        prog_geojson = output_dir / "hourly_fire_progression.geojson"

        gdf_prob.to_file(prob_geojson, driver="GeoJSON")
        gdf_prog.to_file(prog_geojson, driver="GeoJSON")

        print(f"Wrote {prob_geojson}")
        print(f"Wrote {prog_geojson}")

    if write_gpkg:
        gpkg_path = output_dir / "wrfsfire_arcgis_upload.gpkg"

        if gpkg_path.exists():
            gpkg_path.unlink()

        gdf_prob.to_file(
            gpkg_path,
            layer="fire_probability",
            driver="GPKG",
        )

        gdf_prog.to_file(
            gpkg_path,
            layer="hourly_fire_progression",
            driver="GPKG",
        )

        print(f"Wrote {gpkg_path}")

    if write_shapefiles:
        # Shapefile field names must be short. This script already uses short
        # names such as pclass, pmin, pmax, vtime, pthresh.
        write_zipped_shapefile(
            gdf_prob,
            output_dir / "fire_probability_shp.zip",
            "fire_prob",
        )

        write_zipped_shapefile(
            gdf_prog,
            output_dir / "hourly_fire_progression_shp.zip",
            "fire_prog",
        )


def main() -> None:
    args = parse_args()

    input_file = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_file.exists():
        raise FileNotFoundError(input_file)

    print(f"Opening NetCDF: {input_file}")

    ds = xr.open_dataset(
        input_file,
        chunks={
            "ens": 1,
            "Time": 1,
        },
    )

    required = [args.fire_var, args.lon_var, args.lat_var]
    missing = [v for v in required if v not in ds]

    if missing:
        raise ValueError(f"Missing required variables in NetCDF: {missing}")

    if "ens" not in ds[args.fire_var].dims:
        raise ValueError(f"{args.fire_var} must have an ens dimension.")

    if "Time" not in ds[args.fire_var].dims:
        raise ValueError(f"{args.fire_var} must have a Time dimension.")

    print("Building fire probability layer...")

    gdf_prob = build_fire_probability_layer(
        ds=ds,
        fire_var=args.fire_var,
        lon_var=args.lon_var,
        lat_var=args.lat_var,
        threshold=args.threshold,
        probability_mode=args.probability_mode,
        probability_bins=args.probability_bins,
        area_crs=args.area_crs,
        min_area_ha=args.min_area_ha,
    )

    print(f"Fire probability features: {len(gdf_prob)}")

    print("Building hourly fire progression layer...")

    gdf_prog = build_hourly_progression_layer(
        ds=ds,
        fire_var=args.fire_var,
        lon_var=args.lon_var,
        lat_var=args.lat_var,
        threshold=args.threshold,
        progression_mode=args.progression_mode,
        progression_prob_threshold=args.progression_prob_threshold,
        area_crs=args.area_crs,
        min_area_ha=args.min_area_ha,
    )

    print(f"Hourly progression features: {len(gdf_prog)}")

    print("Writing ArcGIS upload files...")

    write_outputs(
        gdf_prob=gdf_prob,
        gdf_prog=gdf_prog,
        output_dir=output_dir,
        write_geojson=not args.skip_geojson,
        write_gpkg=not args.skip_gpkg,
        write_shapefiles=args.write_shapefiles,
    )

    print("Done.")
    print("")
    print("Manual ArcGIS upload recommendation:")
    print("  1. Upload fire_probability.geojson or fire_probability_shp.zip")
    print("  2. Upload hourly_fire_progression.geojson or hourly_fire_progression_shp.zip")
    print("  3. Create hosted feature layers")
    print("  4. Add both layers to a Map Viewer web map")
    print("  5. Style fire_probability by pclass")
    print("  6. Style hourly_fire_progression by hour or vtime")


if __name__ == "__main__":
    main()