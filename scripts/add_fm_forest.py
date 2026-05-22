import netCDF4 as nc
import shutil

shutil.copyfile("wrfinput_d03", "wrfinput_d03_orig")
ds_inp = nc.Dataset("wrfinput_d03", "a")
ds_inp["FMC_GC"][0, 0, :, :] = 0.04
ds_inp["FMC_GC"][0, 1, :, :] = 0.05
ds_inp["FMC_GC"][0, 2, :, :] = 0.06
ds_inp.close()