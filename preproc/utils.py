from scipy.ndimage import distance_transform_edt
from rasterio.features import rasterize
from voronoi import get_building_class
from rasterio.enums import Resampling
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from shapely.geometry import box
import rioxarray as rxr
import geopandas as gpd
import os.path as osp
from glob import glob
import xarray as xr
import pandas as pd
import numpy as np
import json


fuels_metadata = {
    "nfuel_cat": {
        "file_tag": "_SB40.tif",
        "units": "-",
        "description": "Scott & Burgan 40 fire behaviour fuel model",
        "no_data": [0., 32767.]
    },
    "fueldepthm": {
        "file_tag": "_depth.tif",
        "units": "m",
        "description": "surface fuel depth"
    },
    "fmc_gc01": {
        "file_tag": "_moist1.tif",
        "units": "dec",
        "description": "1 hour surface fuel moisture"
    },
    "fmc_gc02": {
        "file_tag": "_moist10.tif",
        "units": "dec",
        "description": "10 hour surface fuel moisture"
    },
    "fmc_gc03": {
        "file_tag": "_moist100.tif",
        "units": "dec",
        "description": "100 hour surface fuel moisture"
    },
    "fuelload_gc01": {
        "file_tag": "_rhof1.tif",
        "units": "kg/m^2",
        "description": "1 hour surface fuel loading"
    },
    "fuelload_gc02": {
        "file_tag": "_rhof10.tif",
        "units": "kg/m^2",
        "description": "10 hour surface fuel loading"
    },
    "fuelload_gc03": {
        "file_tag": "_rhof100.tif",
        "units": "kg/m^2",
        "description": "100 hour surface fuel loading"
    },
    "savr": {
        "file_tag": "_SAV.tif",
        "units": "1/m",
        "description": "Fuel surface area to volume",
        "no_data": [0.]
    }
}


# -----------------------------
# Reading Routines
# -----------------------------
def read_ignitions(path):
    gdf_ign = gpd.read_file(path)
    ign_coords = gdf_ign.geometry.iloc[0].coords[0]
    ign_time = gdf_ign.start.iloc[0]
    return ign_time, ign_coords


def read_weather_csv(path):
    # Read station metadata
    n_header = 6
    with open(path) as f:
        header_lines = [next(f).strip() for _ in range(n_header)]
        
    metadata = {}
    for line in header_lines:
        if ":" in line:
            key, val = line.split(": ", 1)
            key = key.split("# ")[-1]
            val = val.split(",")[0]
            metadata[key.strip()] = val.strip()
            
    # Read station data
    df = pd.read_csv(path, skiprows=n_header)
    # Get units into properties
    df = df.drop(df.index[0])
    # Transform numeric columns
    cols = [
        "air_temp_set_1",
        "relative_humidity_set_1",
        "wind_speed_set_1",
        "wind_gust_set_1",
        "wind_direction_set_1",
    ]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Date_Time"] = pd.to_datetime(df["Date_Time"], errors="coerce")
    df = df.dropna(subset=["Date_Time"])
    return df, metadata


def read_weather_json(path):
    # Open JSON file and convert to pandas
    df = pd.DataFrame(json.load(open(path))["STATION"][0]["OBSERVATIONS"])
    # From default Synoptic units to exercise units
    df.air_temp_set_1 = pd.to_numeric(df.air_temp_set_1, errors="coerce") * 9/5 + 32
    df.relative_humidity_set_1 = pd.to_numeric(df.relative_humidity_set_1, errors="coerce")
    df.wind_speed_set_1 = pd.to_numeric(df.wind_speed_set_1, errors="coerce") * 2.23694
    df.wind_gust_set_1 = pd.to_numeric(df.wind_gust_set_1, errors="coerce") * 2.23694
    df.wind_direction_set_1 = pd.to_numeric(df.wind_direction_set_1, errors="coerce")
    df = df.rename(columns={"date_time": "Date_Time"})
    return df


def read_fuels_data(path, case_id, ign_coords, wx_coords):
    xr_data_arrays = []
    for var_name, var_data in fuels_metadata.items():
        var_units = var_data["units"]
        var_desc  = var_data["description"]
        var_nd    = var_data.get("no_data", [])
        var_fill  = var_data.get("fill_value", None)
        print(f"Processing variable {var_name}")
        var_paths = sorted(glob(osp.join(path, case_id + "*" + var_data["file_tag"])))
        if len(var_paths) == 1:
            var_path = var_paths[0]
            print(f"Data path found {var_path}")
        else:
            print(f"Data not found, skipping {var_name}...")
            continue
        ds = xr.open_dataset(var_path)
        da = ds.band_data[0]
        for val in var_nd:
            da = da.where(da != val)
        if var_fill != None:
            da = da.fillna(var_fill)
        da = da.assign_attrs({
            "var_name": var_name,
            "description": var_desc,
            "units": var_units
        })
        da = da.rename(var_name)
        xr_data_arrays.append(da)
        fig, ax = plt.subplots()
        plot_xr_da(da, ax)
        ax.plot(ign_coords[0], ign_coords[1], "r*")
        ax.plot(wx_coords, "k^")
        plt.show()
    ds = xr.merge(xr_data_arrays)
    return ds


def read_temp_dataset(path):
    ds_temp = rxr.open_rasterio(path)
    return ds_temp


def read_terrain_data(elev_base_path, ds_temp, ign_coords, wx_coords):
    ds_elev = rxr.open_rasterio(elev_base_path)
    ds_elev = ds_elev.rio.reproject_match(ds_temp, resampling=Resampling.bilinear)
    da = ds_elev.isel(band=0)
    da = da.assign_attrs({
        "var_name": "zsf",
        "description": "Terrain Elevation",
        "units": "m"
    })
    da = da.rename("zsf")
    da = da.where(da != -32768.)
    fig, ax = plt.subplots()
    plot_xr_da(da, ax)
    ax.plot(ign_coords[0], ign_coords[1], "r*")
    ax.plot(wx_coords, "k^")
    plt.show()
    return da
    
    
def read_building_data(
        path, lot_length_col="nominal lot length [m]", lot_width_col="nominal lot width [m]", 
        building_length_col="building length [m]", building_width_col="building width [m]", 
        building_height_col="building height [m]"
    ):
    gdf_build = gpd.read_file(path)
    # Process properties required by WUDAPT classification
    gdf_build["building_height"] = gdf_build[building_height_col]
    gdf_build["lot_length"] = (gdf_build[lot_length_col] + gdf_build[lot_width_col]) / 2.
    gdf_build["side_length"] = (gdf_build[building_length_col] + gdf_build[building_width_col]) / 2.
    gdf_build["separation_length"] = gdf_build["lot_length"] - gdf_build["side_length"] 
    return gdf_build


def read_road_data(path):
    gdf_roads = gpd.read_file(path)
    return gdf_roads


def read_tree_data(path):
    gdf_trees = gpd.read_file(path)
    return gdf_trees
    

# -----------------------------
# Processing Routines
# -----------------------------
def define_domain_center(da, shift=[]):
    mask = da.notnull()
    y_valid = mask.any(dim="x")
    x_valid = mask.any(dim="y")
    da_cropped = da.sel(y=y_valid, x=x_valid)
    idx = int(da_cropped.x.shape[0]/2)
    idy = int(da_cropped.y.shape[0]/2)
    if len(shift) == 2:
        idx += shift[0]
        idy += shift[1]
    xc, yc = da_cropped.x[idx].data, da_cropped.y[idy].data
    xc = float(xc)
    yc = float(yc)
    return xc, yc


def rasterize_buildings(gdf_build, ds_temp, coarse_res = 60.0):
    # minimum amount of buildings
    min_buildings = 1

    # Reproject first to a projected CRS (in meters)
    gdf_use = gdf_build.to_crs("EPSG:32610")
    template_da = ds_temp.rio.reproject("EPSG:32610")

    xmin, ymin, xmax, ymax = template_da.rio.bounds()

    xs = np.arange(xmin, xmax + coarse_res, coarse_res)
    ys = np.arange(ymin, ymax + coarse_res, coarse_res)

    cells = []
    ids = []

    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            cells.append(box(xs[i], ys[j], xs[i + 1], ys[j + 1]))
            ids.append(len(ids))

    grid = gpd.GeoDataFrame({"cell_id": ids}, geometry=cells, crs=template_da.rio.crs)

    # Assign each building to a coarse cell using centroid
    bldg_pts = gdf_use.copy()
    bldg_pts["geometry"] = bldg_pts.geometry.centroid

    joined = gpd.sjoin(
        bldg_pts,
        grid[["cell_id", "geometry"]],
        how="left",
        predicate="within"
    )

    stats = (
        joined.dropna(subset=["cell_id"])
        .groupby("cell_id")
        .agg(
            building_height=("building_height", "mean"),
            side_length=("side_length", "mean"),
            separation_length=("separation_length", "mean"),
            n_buildings=("side_length", "size"),
        )
        .reset_index()
    )

    grid_stats = grid.merge(stats, on="cell_id", how="left")

    # mask poorly sampled cells
    for col in ["building_height", "side_length", "separation_length"]:
        grid_stats.loc[
            grid_stats["n_buildings"].fillna(0) < min_buildings,
            col
        ] = np.nan

    # Rasterize coarse cells directly onto the fine template grid
    transform = template_da.rio.transform()
    out_shape = (template_da.sizes["y"], template_da.sizes["x"])

    layers = {}

    for col in ["building_height", "side_length", "separation_length"]:
        shapes = [
            (geom, val)
            for geom, val in zip(grid_stats.geometry, grid_stats[col])
            if np.isfinite(val)
        ]

        arr = rasterize(
            shapes,
            out_shape=out_shape,
            transform=transform,
            fill=np.nan,
            dtype="float32",
            all_touched=True,
        )
        
        arr[~np.isfinite(arr)] = 0.

        layers[col] = xr.DataArray(
            arr,
            dims=("y", "x"),
            coords={
                "y": template_da.y,
                "x": template_da.x,
                "spatial_ref": template_da.spatial_ref,
            },
            name=col,
            attrs={"units": "m"},
        ).rio.write_crs(template_da.rio.crs)

        ds_building_props = xr.Dataset(layers)
        ds_building_props = ds_building_props.rio.reproject_match(
            ds_temp,
            resampling=Resampling.bilinear
        )
    return ds_building_props


def get_wudapt_classes(ds):
    wudapt_classes = []
    for q_side, q_sep in zip(ds["side_length"].data.ravel(), ds["separation_length"].data.ravel()):
        if q_side == 0. or q_sep == 0.:
            wudapt_classes.append(0.)
            continue
        _, idx = get_building_class(q_side, q_sep, return_distance=False)
        wudapt_classes.append(idx + 1) 
    wudapt_classes = np.reshape(wudapt_classes, ds["side_length"].shape)
    return wudapt_classes


def add_variable_to_ds(ds, var_data, var_name):
    da = xr.DataArray(
        var_data,
        dims=("y", "x"),
        coords={
            "y": ds.y,
            "x": ds.x,
            "spatial_ref": ds.spatial_ref,
        },
        name=var_name
    )
    da = da.rio.write_crs(ds.rio.crs)
    da = da.rio.write_transform(ds.rio.transform())
    ds[var_name] = da
    return ds


def rasterize_roads(gdf_roads, ds_temp, road_buffer = 10.0):
    # Reproject first to a projected CRS (in meters)
    roads = gdf_roads.to_crs("EPSG:32610")
    template_da = ds_temp.rio.reproject("EPSG:32610")

    # Match CRS
    if roads.crs != template_da.rio.crs:
        roads = roads.to_crs(template_da.rio.crs)

    transform = template_da.rio.transform()
    out_shape = (template_da.sizes["y"], template_da.sizes["x"])

    # Optional: buffer roads so thin lines appear on the grid
    roads["geometry"] = roads.geometry.buffer(road_buffer)

    transform = template_da.rio.transform()
    out_shape = (template_da.sizes["y"], template_da.sizes["x"])

    shapes = [
        (geom, 1)
        for geom in roads.geometry
        if geom is not None and not geom.is_empty
    ]

    road_mask = rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    )

    road_da = xr.DataArray(
        road_mask,
        dims=("y", "x"),
        coords={
            "y": template_da.y,
            "x": template_da.x,
            "spatial_ref": template_da.spatial_ref,
        },
        name="road_mask",
    )
    road_da = road_da.rio.write_crs(template_da.rio.crs)
    road_da = road_da.rio.write_transform(transform)

    road_da = road_da.rio.reproject_match(
        ds_temp,
        resampling=Resampling.nearest
    )
    road_da = road_da.astype(bool)
    return road_da


def rasterize_trees(gdf_trees, ds_temp):
    # Match CRS
    trees = gdf_trees.copy()
    if trees.crs is None:
        trees = trees.set_crs("EPSG:4326")

    if trees.crs != ds_temp.rio.crs:
        trees = trees.to_crs(ds_temp.rio.crs)

    transform = ds_temp.rio.transform()
    out_shape = (ds_temp.sizes["y"], ds_temp.sizes["x"])

    vars_to_rasterize = {
        "HT":  {"out_name": "canopy_height",       "units": "m"},
        "CBH": {"out_name": "canopy_base_height",  "units": "m"},
        "CBD": {"out_name": "canopy_bulk_density", "units": "kg m-3"},
    }

    layers = {}

    for col, meta in vars_to_rasterize.items():
        shapes = [
            (geom, float(val))
            for geom, val in zip(trees.geometry, trees[col])
            if geom is not None
            and not geom.is_empty
            and np.isfinite(val)
        ]
        
        arr = rasterize(
            shapes=shapes,
            out_shape=out_shape,
            transform=transform,
            fill=np.nan,
            dtype="float32",
            all_touched=True,
        )
        
        # Replace NaN values with 0
        arr[np.isfinite(arr) == False] = 0.

        da = xr.DataArray(
            arr,
            dims=("y", "x"),
            coords={
                "y": ds_temp.y,
                "x": ds_temp.x,
                "spatial_ref": ds_temp.spatial_ref,
            },
            name=meta["out_name"],
            attrs={
                "source_column": col,
                "units": meta["units"],
            },
        )

        da = da.rio.write_crs(ds_temp.rio.crs, inplace=False)
        da = da.rio.write_transform(transform, inplace=False)

        layers[meta["out_name"]] = da

    ds_trees = xr.Dataset(layers)

    return ds_trees

# -----------------------------
# Verification Routines
# -----------------------------
def check_fuel_props(ds):
    # summarize per nfuel_cat the associated fuel properties
    nfuel_arr = ds["nfuel_cat"].values
    cat_vals = np.unique(nfuel_arr[np.isfinite(nfuel_arr)])
    # coerce to ints if categories are integer-like
    if np.all(np.mod(cat_vals, 1) == 0):
        cat_vals = cat_vals.astype(int)

    vars_to_check = ["fueldepthm", "fuelload_gc01", "fuelload_gc02", "fuelload_gc03", "savr"]
    rows = []
    for v in cat_vals:
        mask = np.isfinite(nfuel_arr) & (nfuel_arr == v)
        row = {"nfuel_cat": int(v) if np.isfinite(v) and float(v).is_integer() else float(v), "count": int(mask.sum())}
        for var in vars_to_check:
            if var not in ds:
                row[var] = None
                continue
            arr = ds[var].values
            vals = np.unique(arr[mask])
            vals = vals[np.isfinite(vals)]
            if len(vals) > 1:
                vals = vals[vals != 0.]
                if len(vals) > 1:
                    print(f"Warning: multiple values found for nfuel_cat {v} and variable {var}: {vals}")
            row[var] = np.sort(vals).tolist()[0] if vals.size else None
        rows.append(row)

    df_nfuel_summary = pd.DataFrame(rows).sort_values("nfuel_cat").reset_index(drop=True)
    return df_nfuel_summary


def define_nfuel_cat_urban(ds, wudapt_classes, road_da):
    # Add WUDAPT classes to fuel classes map
    nfuel_cat_urban = ds["nfuel_cat"].data.copy()
    urban_mask = np.isfinite(wudapt_classes) & ~np.isfinite(nfuel_cat_urban)
    nfuel_cat_urban[urban_mask] = -wudapt_classes[urban_mask]
    # True where roads are
    road = road_da.values.astype(bool)
    # Pixels that already have a valid fuel class
    valid = np.isfinite(nfuel_cat_urban)
    # For every pixel, find nearest valid fuel pixel
    _, nearest = distance_transform_edt(
        ~valid,
        return_distances=True,
        return_indices=True
    )
    # Target pixels
    target = road & ~valid
    # Fill only road pixels
    nfuel_cat_urban[target] = nfuel_cat_urban[
        nearest[0][target],
        nearest[1][target]
    ]
    # Unknowns set to unburnable
    nfuel_cat_urban[~np.isfinite(nfuel_cat_urban)] = 0.
    # Add variable into final dataset
    ds["nfuel_cat_urban"] = xr.DataArray(
        nfuel_cat_urban,
        dims=("y", "x"),
        coords={
            "y": ds.y,
            "x": ds.x,
            "spatial_ref": ds.spatial_ref,
        },
        name="nfuel_cat_urban",
    )
    ds["nfuel_cat_urban"] = ds["nfuel_cat_urban"].rio.write_crs(ds.rio.crs)
    ds["nfuel_cat_urban"] = ds["nfuel_cat_urban"].rio.write_transform(ds.rio.transform())
    return ds


# -----------------------------
# Plotting Routines
# -----------------------------
def plot_xr_da(da, ax=None):
    # Create map plot
    da.plot.pcolormesh(ax=ax)
    # Print unique values
    print(da.attrs["var_name"], np.unique(da.data))
    ax.set_title(da.attrs["description"])
    return ax


def plot_weather_data(df, date_time_row="Date_Time"):
    fig, ax_temp = plt.subplots(figsize=(15, 7))
    fig.subplots_adjust(right=0.82)
    # Additional y-axes
    ax_rh = ax_temp.twinx()
    ax_ws = ax_temp.twinx()
    ax_wg = ax_temp.twinx()
    # Offset the extra right-side axes
    ax_ws.spines["right"].set_position(("axes", 1.10))
    ax_wg.spines["right"].set_position(("axes", 1.20))
    # Make offset axes visible
    ax_ws.spines["right"].set_visible(True)
    ax_wg.spines["right"].set_visible(True)

    # Plot each variable
    l1, = ax_temp.plot(
        df[date_time_row], df["air_temp_set_1"],
        color="red", lw=2.0, label="Air Temp"
    )
    l2, = ax_rh.plot(
        df[date_time_row], df["relative_humidity_set_1"],
        color="blue", lw=2.0, label="RH"
    )
    l3, = ax_ws.plot(
        df[date_time_row], df["wind_speed_set_1"],
        color="black", lw=2.0, label="Wind Speed"
    )
    l4, = ax_wg.plot(
        df[date_time_row], df["wind_gust_set_1"],
        color="purple", lw=1.8, ls="--", label="Wind Gust"
    )

    # Axis labels and styling
    ax_temp.set_ylabel("Air Temp (F)", color="red")
    ax_rh.set_ylabel("RH (%)", color="blue")
    ax_ws.set_ylabel("Wind Speed (mph)", color="black")
    ax_wg.set_ylabel("Wind Gust (mph)", color="purple")

    ax_temp.tick_params(axis="y", colors="red")
    ax_rh.tick_params(axis="y", colors="blue")
    ax_ws.tick_params(axis="y", colors="black")
    ax_wg.tick_params(axis="y", colors="purple")

    ax_temp.spines["left"].set_color("red")
    ax_rh.spines["right"].set_color("blue")
    ax_ws.spines["right"].set_color("black")
    ax_wg.spines["right"].set_color("purple")

    ax_temp.grid(True, alpha=0.25)
    ax_temp.set_xlabel("Date Time")

    # Time axis formatting
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax_temp.xaxis.set_major_locator(locator)
    ax_temp.xaxis.set_major_formatter(formatter)

    # Combined legend
    lines = [l1, l2, l3, l4]
    labels = [ln.get_label() for ln in lines]
    ax_temp.legend(lines, labels, loc="upper left", frameon=True)

    station_id = df["Station_ID"].iloc[0] if "Station_ID" in df.columns and len(df) else "Unknown"
    ax_temp.set_title(f"Station {station_id}, Meteorological Time Series")
    plt.show()


def plot_wind_rose(df):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="polar")
    wd = df["wind_direction_set_1"].to_numpy()
    ws = df["wind_speed_set_1"].to_numpy()
    mask = np.isfinite(wd) & np.isfinite(ws)
    wd = wd[mask]
    ws = ws[mask]
    # Convert direction to radians
    theta = np.deg2rad(wd)
    # Direction bins or sectors
    nsector = 32
    dir_edges = np.linspace(0, 2 * np.pi, nsector + 1)
    dir_centers = 0.5 * (dir_edges[:-1] + dir_edges[1:])
    dir_width = dir_edges[1] - dir_edges[0]
    # Speed bins
    speed_bins = [0, 10, 15, 18, 20, 22, 24, 26, 28, 30, np.inf]
    speed_labels = [
        "0-10", "10-15", "15-18", "18-20", "20-22",
        "22-24", "24-26", "26-28", "28-30", "30+"
    ]
    # Histogram by direction and speed
    heights_by_speed = []
    for lo, hi in zip(speed_bins[:-1], speed_bins[1:]):
        h = np.zeros(nsector)
        sel_speed = (ws >= lo) & (ws < hi)
        counts, _ = np.histogram(theta[sel_speed], bins=dir_edges)
        h[:] = counts
        heights_by_speed.append(h)
    # Stack bars
    bottom = np.zeros(nsector)
    rose_colors = ["#d9d9d9", "#9ecae1", "#6baed6", "#3182bd", "#08519c"]
    for h, c, lab in zip(heights_by_speed, rose_colors, speed_labels):
        ax.bar(
            dir_centers,
            h,
            width=dir_width,
            bottom=bottom,
            color=c,
            edgecolor="white",
            linewidth=0.5,
            align="center",
            label=lab
        )
        bottom += h
    # Polar formatting
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    ax.set_title("Wind Rose", va="bottom", fontsize=10)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.35),
        ncol=3,
        fontsize=8,
        frameon=True
    )
    plt.show()
    
    
def plot_building_rasters(ds_building_props):
    ds_building_props["building_height"].plot()
    plt.title("Building Height (m)")
    plt.show()
    ds_building_props["side_length"].plot()
    plt.title("Side Length (m)")
    plt.show()
    ds_building_props["separation_length"].plot()
    plt.title("Separation Length (m)")
    plt.show()