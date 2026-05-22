####### WORKFLOW FOR TOWN OF FORESTS #######
# Notes:
#   1- Every numbered step needs to wait for the previous step
#   2- This workflow assumes all the data preparation is done

#------------------------------
# 1) Run SIM_BASELINE
#------------------------------
# This is using WRFxPy configuration
cd wrfxpy
./forecast.sh jobs/forest.json >& logs/forest.log

#------------------------------
# 2) Run SIM_CONTROL
#------------------------------
# Before re-running the job: 
#   1- Rename the previous folder to not overwrite 
#   2- Modify etc/vtables/geo_vars.json to point to the new GeoTIFF files
#   3- Add flag `run_wrf: False` in forest.json
#   4- Change etc/nlists/default.input to the one with flag `use_urban_module = 1`
./forecast.sh jobs/forest.json >& logs/forest_prep.log
cd wksp/wfc-forest-control-2024-10-27_16:00:00-13/wrf
ln -s ../../../../scripts/add_fm_forest.py .
python add_fm_forest.py
cd ../../../
./execute_wrf.sh wfc-forest-control-2024-10-27_16:00:00-13 >& logs/forest_wrf.log

#------------------------------
# 3) Prepare template
#------------------------------
cd wksp/wfc-forest-control-2024-10-27_16:00:00-13
mkdir wrf_ens
cd wrf
cp -ra $(ls * | grep -v wrfout | grep -v wrfrst | grep -v rsl) ../wrf_template
cd ../../../../../

#------------------------------
# 4) Run SIM_ENSEMBLE
#------------------------------
cd scripts
python run_wrf_ensemble.py \
	--template-run-dir ../wrfxpy/wksp/wfc-forest-control-2024-10-27_16:00:00-13/wrf_template/ \
	--ensemble-root ../wrfxpy/wksp/wfc-forest-control-2024-10-27_16:00:00-13/wrf_ens \
	--n-members 10 --cores-per-run 42 --runs-per-job 2
	
#------------------------------
# Run postprocessing pipeline
#------------------------------
# Postproc SIM-BASELINE
python postproc_sims.py \
	--ensemble-dir ../wrfxpy/wksp/wfc-forest-control-2024-10-27_16:00:00-13_base \
	--output ../data/output/model/Forest_SIM_BASELINE.nc --member-glob "wrf" \
	--wrfout-glob "wrfout_d03*" --vars FXLONG FXLAT NFUEL_CAT FIRE_AREA \
	--static-vars FXLONG FXLAT NFUEL_CAT
# Postproc SIM-CONTROL
python postproc_sims.py \
	--ensemble-dir ../wrfxpy/wksp/wfc-forest-control-2024-10-27_16:00:00-13 \
	--output ../data/output/model/Forest_SIM_CONTROL.nc --member-glob "wrf" \
	--wrfout-glob "wrfout_d03*" --vars FXLONG FXLAT NFUEL_CAT FIRE_AREA \
	--static-vars FXLONG FXLAT NFUEL_CAT
# Postporc SIM-ENSEMBLE
python postproc_sims.py \
	--ensemble-dir ../wrfxpy/wksp/wfc-forest-control-2024-10-27_16:00:00-13/wrf_ens \
	--output ../data/output/model/Forest_SIM_ENSEMBLE.nc --wrfout-glob "wrfout_d03*" \
	--vars FXLONG FXLAT NFUEL_CAT FIRE_AREA --static-vars FXLONG FXLAT NFUEL_CAT
# Create AGOL final products
python postproc_agol.py --input ../data/output/model/Forest_SIM_ENSEMBLE.nc \
	--output-dir ../data/output/agol/Forest --threshold 0 --probability-mode any_time \
	--progression-mode cumulative --progression-prob-threshold 0.5 --min-area-ha 0.1 --write-shapefiles
