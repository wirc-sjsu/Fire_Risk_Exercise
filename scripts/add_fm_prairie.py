from scipy.interpolate import griddata
import netCDF4 as nc
import numpy as np
import shutil

shutil.copyfile("wrfinput_d03", "wrfinput_d03_orig")
ds_inp = nc.Dataset("wrfinput_d03", "a")
srx = 50
sry = 50
lons = ds_inp["XLONG"][0]
lats = ds_inp["XLAT"][0]
flons = ds_inp["FXLONG"][0, :-sry, :-srx]
print(flons)
flats = ds_inp["FXLAT"][0, :-sry, :-srx]
fmc_gc_f = ds_inp["FMC_GC_F"][:].copy()
nfuelcat = ds_inp["NFUEL_CAT"][0].copy()
fm1 = fmc_gc_f[0, 0].copy()
print(fm1.shape, nfuelcat.shape)
fm1[nfuelcat == 30.] = 0.25
fm1[nfuelcat != 30.] = 0.04
fmc_gc_f[0, 0, :, :] = fm1[np.newaxis, np.newaxis, :, :]
fmc_gc_f[0, 1, :, :] = 0.05
fmc_gc_f[0, 2, :, :] = 0.06
ds_inp["FMC_GC_F"][:] = fmc_gc_f 
coords = np.c_[flons.ravel(), flats.ravel()]
vals = fmc_gc_f[0, 0, :-sry, :-srx].ravel()
xi = np.c_[lons.ravel(), lats.ravel()]
print(coords.shape, vals.shape)
fmc_gc_1 = griddata(coords, vals, xi)
fmc_gc_1 = np.reshape(fmc_gc_1, lons.shape)
ds_inp["FMC_GC"][0, 0, :, :] = fmc_gc_1
ds_inp["FMC_GC"][0, 1, :, :] = 0.05
ds_inp["FMC_GC"][0, 2, :, :] = 0.06
ds_inp.close()