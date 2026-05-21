# Wildfire Risk Modeling Exercise
This repository contains a submission to the [Wildfire Risk Modeling Exercise](https://www.wildfirecommons.org/educationhub/datachallenge/learner/4da8e5b0-6d81-4aa9-9482-309194cfb975) by the [Wildfire Interdisciplinary Research Center](https://www.wildfirecenter.org/) (WIRC) at San José State University (SJSU). The submission uses WRF-SFIRE ensemble wildfire simulations to estimate fire probability and derive additional diagnostics from the coupled atmosphere-fire model.


## Prerequisites
- [WRF-SFIRE](https://github.com/openwfm/WRF-SFIRE)
- [WRFxPy](https://github.com/openwfm/WRFxPy)
- [Python](https://www.python.org/)
- [SciPy](https://scipy.org/)
- [Rasterio](https://rasterio.readthedocs.io/)
- [Xarray](https://docs.xarray.dev/)
- [RioXarray](https://corteva.github.io/rioxarray/)
- [GeoPandas](https://geopandas.org/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)


## Wildfire Commons Resources
- **Forest landscape data.** The Wildfire Commons catalogs landscape, building, and road data (in `.tif` and `.geojson` format).
- **Ignition coordinates.** The given latitude/longitude coordinate and time for:
     - **Forest**: [38.9014, -120.0306] on 2024-10-27 at 1700 UTC.
     - **Prairie**: [35.24435, -101.9262636] on 2026-03-09 at 1500 UTC.
- **Weather parameters.** Date and times to acquire HRRR weather forcing.


## Data Preparation
**`data_preparation_forest.ipynb`** and **`data_preparation_prairie.ipynb`**
- Read ignition information
- Compare synthetic weather data to the real Synoptic weather station for the same date-time period
- Visualize weather data
- Visualize fuel data
- Check that the fuel properties match the namelist options in WRF-SFIRE
- Visualize terrain data
- Define the center of the domain
- Rasterize building data into building side length, separation length, and height
- Classify areas into WUDAPT classes based on side length and separation length
- Rasterize roads into a mask
- Interpolate roads to dominant fuel classes
- Combine SB40, WUDAPT, and road interpolation into fuel data for WRF-SFIRE


## WRFxPy Forecasting System Processing
**`github.com/openwfm/wrfxpy`**
- TODO


## WRF-SFIRE WUI Fire Spread Simulation
**`github.com/openwfm/WRF-SFIRE`**
- TODO


## WRF-SFIRE Urban Fire Spread Simulation
**`github.com/openwfm/WRF-SFIRE`**
- TODO


# Final Model Outputs
**`outputs/`** directory
- TODO

