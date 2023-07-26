#!/usr/bin/python
## P.Espin 
### PLOT time series and change the file of GPS to LOS
### Jess Payne update 19/05/23:
# Equation to calculate LoS GNSS displacements from three component GNSS displacements (Fialko & Simons, 2001;
# Stephens et al, 2020)
# The below script takes a .tenv3 three component GNSS file as downloaded from Nevada Geodetic GPS Portal
# and convert the three components into LoS direction to compare to LoS displacments calculated using
# Sentinel-1 InSAR.
# Linear fits to GNSS and InSAR data are calculated and plotted.
# .tenv3 parameter extraction and linear vertical GNSS fit translated from John Elliott MATLAB script
#
# InSAR data is calculated using LiCS processing tools.
#
# Inputs required:
# 1. .h5/.nc file as output from LiCSBAS or licsar2licsbas
# 1b. Parameter file as output from LiCSBAS or licsar2licsbas (noramlly output as EQA.dem_par)
# 2. .tenv3 file downloaded from http://geodesy.unr.edu/NGLStationPages/gpsnetmap/GPSNetMap.html
# 3. Lat, lon and name of GNSS site (find on website in 2.)
# 4. InSAR LiCS frame name

# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.transforms as mtransforms
import math
import netCDF4
from scipy.stats import linregress
from scipy.interpolate import griddata
import h5py
from datetime import datetime
from cmcrameri import cm
import xarray as xr
import imageio.v2 as imageio

# this is Andrew Watson's library of functions, see https://github.com/Active-Tectonics-Leeds/interseismic_practical
import sys
sys.path.append('/nfs/a285/homes/eejap/plots/gnss_insar_ts')
import interseis_lib as lib


#%%
#Directory path
directorio= '/nfs/a285/homes/eejap/plots/gnss_insar_ts'
data_directory = '/nfs/a285/homes/eejap/reunwrap_tests'
gnss_station = "MMX1"
insar_frame = "143D_07197_212120"
ml_clip = "ml1_clip" # ml10 or ml1_clip

filename = directorio + '/data/gnss/' + gnss_station + '.NA.tenv3'
gnss_station_info = directorio + '/data/gnss/gnss_info.txt'
formatSpec = '%4C%8s%10f%6f%5f%2f%7f%7f%10f%10f%10f%6f%10f%8f%9f%9f%9f%10f%10f%f%[^\n\r]'
#nc_file = '/nfs/a285/homes/eejap/reunwrap_tests/078A_07049_131313/gucmT_ml1_clip/078A_07049_131313_gold_casc_ml1_clip.nc'
# h5_file_rw = directorio + '/data/' + insar_frame + '/' + insar_frame + '_gucmT_' + ml_clip +'.cum.h5'
# h5_file_def = directorio + '/data/' + insar_frame + '/' + insar_frame + '_' + ml_clip +'.cum.h5'
# par_file_rw = directorio + '/data/' + insar_frame + '/' + insar_frame + '_gucmT_' + ml_clip +'.dem_par'
# par_file_def = directorio + '/data/' + insar_frame + '/' + insar_frame + '_' + ml_clip +'.dem_par'
# LOSufile = directorio + '/data/' + insar_frame + '/' + insar_frame + '_' + ml_clip +'.U.geo'
# LOSefile = directorio + '/data/' + insar_frame + '/' + insar_frame + '_' + ml_clip +'.E.geo'
# LOSnfile = directorio + '/data/' + insar_frame + '/' + insar_frame + '_' + ml_clip +'.N.geo'

h5_file_rw = data_directory + '/' + insar_frame + '/' + insar_frame + '_gucmT_' + ml_clip +'.cum.h5'
h5_file_def = data_directory + '/' + insar_frame + '/' + insar_frame + '_' + ml_clip +'.cum.h5'
par_file_rw = data_directory + '/' + insar_frame + '/' + insar_frame + '_gucmT_' + ml_clip +'.dem_par'
par_file_def = data_directory + '/' + insar_frame + '/' + insar_frame + '_' + ml_clip +'.dem_par'
LOSufile = data_directory + '/' + insar_frame + '/' + insar_frame + '_' + ml_clip +'.U.geo'
LOSefile = data_directory + '/' + insar_frame + '/' + insar_frame + '_' + ml_clip +'.E.geo'
LOSnfile = data_directory + '/' + insar_frame + '/' + insar_frame + '_' + ml_clip +'.N.geo'
vel_def_filt = data_directory + '/' + insar_frame + '/' + insar_frame + '_' + ml_clip +'.vel.filt.geo.tif'
# load in filtered cum.h5 to compare t-s

#%%
dfgnss = pd.read_csv(filename, delimiter=r"\s+")
dfgnss_info = pd.read_csv(gnss_station_info, delimiter=r"\s+")
dfgnss_info.set_index('name', inplace=True)
# nc_dataset = netCDF4.Dataset(nc_file)
# # #%% View parameters of InSAR data in netCDF file
# print('Dimensions:')
# for dim in nc_dataset.dimensions:
#     print(dim, len(nc_dataset.dimensions[dim]))
# #%%
# print('Variables:')
# for var in nc_dataset.variables:
#     print(var, nc_dataset.variables[var].shape)

#%% make vel_def_filt a tif not string
vel_def_filt_arr = imageio.imread(vel_def_filt)
#vel_def_fault_arr = gdal.Open(vel_def_filt)

#%%
gnss_lat = dfgnss_info.at[gnss_station, 'lat']
gnss_lon = dfgnss_info.at[gnss_station, 'lon']
start_date = dfgnss_info.at[gnss_station, 'start_date'].astype(str)
end_date = dfgnss_info.at[gnss_station, 'end_date'].astype(str)
#%% Reformat to decimal year, E, N, U, remove mean and rescale to mm
dfdata = pd.DataFrame({
    'Column2': dfgnss.iloc[:, 2],
    'Column8': dfgnss.iloc[:, 8],
    'Column10': dfgnss.iloc[:, 10],
    'Column12': dfgnss.iloc[:, 12],
    'Column14': dfgnss.iloc[:, 14],
    'Column15': dfgnss.iloc[:, 15],
    'Column16': dfgnss.iloc[:, 16]
})
#%%
dfdata_array = np.squeeze(np.array(dfdata))
mean_value = np.mean(dfdata_array, axis=0).squeeze().reshape(1,-1)
binary_array = np.array([0, 1, 1, 1, 0, 0, 0])
scaling_array = np.array([1, 1000, 1000, 1000, 1000, 1000, 1000])

mean_value = np.tile(mean_value, (dfdata_array.shape[0], 1))
binary_array = np.tile(binary_array, (dfdata_array.shape[0], 1))
scaling_array = np.tile(scaling_array, (dfdata_array.shape[0], 1))

dfdata_array = scaling_array * (dfdata_array - binary_array * mean_value)
ndays = len(dfdata_array)
date = dfdata_array[:, 0]
E = dfdata_array[:, 1]
N = dfdata_array[:, 2]
U = dfdata_array[:, 3]
eE = dfdata_array[:, 4]
eN = dfdata_array[:, 5]
eU = dfdata_array[:, 6]
wE = 1 / eE ** 2
wN = 1 / eN ** 2
wU = 1 / eU ** 2
jday = dfgnss.iloc[:, 3]

column_names = ['Dates', 'dN', 'dE', 'dU', 'Sn', 'Se', 'Su']
dfGPS=pd.DataFrame(dfdata_array, columns=column_names)
#%% get imdates
with h5py.File(h5_file_rw, 'r') as file:
    imdates_rw = file['imdates']
    imdates_rw = imdates_rw[:]  
    vel_rw = file['vel']
    vel_rw = vel_rw[:]
    cum_rw = file['cum']
    cum_rw = cum_rw[:]
    
with h5py.File(h5_file_def, 'r') as file:
    imdates_def = file['imdates']
    imdates_def = imdates_def[:]  
    vel_def = file['vel']
    vel_def = vel_def[:]
    cum_def = file['cum']
    cum_def= cum_def[:]

#%% complete using h5 file
# read array dimensions from par file
width_rw = int(lib.get_par(par_file_rw,'width'))
length_rw = int(lib.get_par(par_file_rw,'nlines'))

width_def = int(lib.get_par(par_file_def,'width'))
length_def = int(lib.get_par(par_file_def,'nlines'))

# get corner positions
corner_lat_rw = float(lib.get_par(par_file_rw, 'corner_lat'))
corner_lon_rw = float(lib.get_par(par_file_rw,'corner_lon'))

corner_lat_def = float(lib.get_par(par_file_def,'corner_lat'))
corner_lon_def = float(lib.get_par(par_file_def,'corner_lon'))

# get post spacing (distance between velocity measurements)
post_lat_rw = float(lib.get_par(par_file_rw,'post_lat'))
post_lon_rw = float(lib.get_par(par_file_rw,'post_lon'))

post_lat_def = float(lib.get_par(par_file_def,'post_lat'))
post_lon_def = float(lib.get_par(par_file_def,'post_lon'))

# calculate grid spacings
lat_rw = corner_lat_rw + post_lat_rw*np.arange(1,length_rw+1) - post_lat_rw/2
lon_rw = corner_lon_rw + post_lon_rw*np.arange(1,width_rw+1) - post_lon_rw/2

lat_def = corner_lat_def + post_lat_def*np.arange(1,length_def+1) - post_lat_def/2
lon_def = corner_lon_def + post_lon_def*np.arange(1,width_def+1) - post_lon_def/2

# #%% Create grid of 0.1 degrees over frame for input into UNAVCO plate motion calculator
# # # make lon lat arrays of spacing 0.1 or 0.05 (0.1 for very coarse grid. If use 0.1 divide length_rw and width_rw below by 10, 0.05 by 2)
# post_lat_grid = -0.05
# post_lon_grid = 0.05
# ref = 'NNR'

# lat_grid = np.round(corner_lat_rw + post_lat_grid*np.arange(1,(length_rw/5)+1) - post_lat_grid/2,6) # divide length_rw by 10 if post lat grid = 0.1; 5 if 0.05
# lon_grid = np.round(corner_lon_rw + post_lon_grid*np.arange(1,(width_rw/5)+1) - post_lon_grid/2,6) # divide length_rw by 10 if post lat grid = 0.1; 5 if 0.05

# lat_grid_2 = corner_lat_rw + (post_lat_grid/5)*np.arange(1,(length_rw)+1) - post_lat_grid/2 # make finer grid for plotting
# lon_grid_2 = corner_lon_rw + (post_lon_grid/5)*np.arange(1,(width_rw)+1) - post_lon_grid/2 # make finer grid for plotting

# # make grided x, y pairs
# coords_grid = []
# for i in range(len(lon_grid)):
#     for j in range(len(lat_grid)):
#         # Create coordinate pairs
#         coords_grid.append((lon_grid[i], lat_grid[j], '0,'))

# # Convert the list to a numpy array
# coords_grid = np.array(coords_grid)

# np.savetxt('./outputs/{}/grids/coords_grid_{}_{}_{}.csv'.format(insar_frame, post_lon_grid, insar_frame, ml_clip), coords_grid, delimiter=' ', fmt='%s')


# #%% plot plate motions
# # Read the data from the file

# data = np.genfromtxt('{}/outputs/{}/grids/plate_motion_{}_{}_{}_ITRF2014_NA_{}.txt'.format(directorio, insar_frame, post_lon_grid, insar_frame, ml_clip, ref), skip_header=1, usecols=(0, 1, 2, 3))

# # Extract the longitude, latitude, and elevation values
# lon = data[:, 0]
# lat = data[:, 1]
# Evel = data[:, 2]
# Nvel = data[:, 3]

# # create grid of x and grid of y
# grid_lon, grid_lat = np.meshgrid(lon_grid_2, lat_grid_2)

# # Interpolate the elevation values onto the grid
# grid_Evel = griddata((lon, lat), Evel, (grid_lon, grid_lat), method='cubic')
# grid_Nvel = griddata((lon, lat), Nvel, (grid_lon, grid_lat), method='cubic')

# # project into LoS
# LOSe = np.fromfile(LOSefile, dtype='float32').reshape((length_def, width_def))
# LOSn = np.fromfile(LOSnfile, dtype='float32').reshape((length_def, width_def))
# plate_los = (grid_Evel*LOSe) + (grid_Nvel*LOSn)
# #%%
# #Plot the raster
# #Create a figure with subplots
# vmin = -100
# vmax = 100
# fig, axs = plt.subplots(1, 3, figsize=(10, 2.5))
# im1 = axs[0].imshow(grid_Evel, extent=(lon_grid_2.min(), lon_grid_2.max(), lat_grid_2.min(), lat_grid_2.max()), cmap = cm.vik)
# im2 = axs[1].imshow(grid_Nvel, extent=(lon_grid_2.min(), lon_grid_2.max(), lat_grid_2.min(), lat_grid_2.max()), cmap = cm.vik)
# im3 = axs[2].imshow(plate_los, extent=(lon_grid_2.min(), lon_grid_2.max(), lat_grid_2.min(), lat_grid_2.max()), cmap = cm.vik)
# cbar = fig.colorbar(im1, ax=axs[0], orientation='vertical', shrink=0.6)
# cbar.set_label('E vel (mm/yr)')
# cbar = fig.colorbar(im2, ax=axs[1], orientation='vertical', shrink=0.6)
# cbar.set_label('N vel (mm/yr)')
# cbar = fig.colorbar(im3, ax=axs[2], orientation='vertical', shrink=0.6)
# cbar.set_label('LOS vel (mm/yr)')
# fig.suptitle('Horizontal Plate Motion, ITRF 2014, rel to {}, {} Spacing'.format(ref, post_lon_grid), y = 1)
# # Adjust the spacing between subplots
# plt.subplots_adjust(wspace=0.4)
# plt.savefig('./outputs/{}/grids/{}_E_N_plate_motion_{}_ITRF2014_{}.jpg'.format(insar_frame, insar_frame, post_lon_grid, ref), dpi=400, bbox_inches='tight')
# plt.show()
# #%%
# # find reference point
# # Find the index of the element with the minimum absolute value
# min_ind_def = np.unravel_index(np.nanargmin(np.abs(vel_def)), vel_def.shape)
# ref_yind_def, ref_xind_def = min_ind_def

# min_ind_rw = np.unravel_index(np.nanargmin(np.abs(vel_rw)), vel_rw.shape)
# ref_yind_rw, ref_xind_rw = min_ind_rw

# # Subtract the value of the minimum element from all elements in the vel array
# vel_def -= vel_def[ref_yind_def, ref_xind_def]
# vel_rw -= vel_rw[ref_yind_rw, ref_xind_rw]

# # Subtract the value of the minimum element from all elements in the plate_los array
# plate_los -= plate_los[ref_yind_def, ref_xind_def]

# # remove plate motion from velocities
# vel_def_plate_motion_corr = vel_def - plate_los
# # add plate motion correction to reunwrapped
# vel_rw_plate_los = vel_rw + plate_los
# #%%
# fig, axs = plt.subplots(2, 2, figsize=(7, 5))
# im1 = axs[0,0].imshow(vel_def, extent=(lon_grid_2.min(), lon_grid_2.max(), lat_grid_2.min(), lat_grid_2.max()), cmap = cm.vik, vmin = vmin, vmax = vmax)
# im2 = axs[1,0].imshow(vel_rw, extent=(lon_grid_2.min(), lon_grid_2.max(), lat_grid_2.min(), lat_grid_2.max()), cmap = cm.vik, vmin = vmin, vmax = vmax)
# im3 = axs[1,1].imshow(vel_def_plate_motion_corr, extent=(lon_grid_2.min(), lon_grid_2.max(), lat_grid_2.min(), lat_grid_2.max()), cmap = cm.vik, vmin = vmin, vmax = vmax)
# im4 = axs[0,1].imshow(vel_rw_plate_los, extent=(lon_grid_2.min(), lon_grid_2.max(), lat_grid_2.min(), lat_grid_2.max()), cmap = cm.vik, vmin = vmin, vmax = vmax)
# cbar = fig.colorbar(im1, ax=axs[0,0], orientation='vertical', shrink=0.8)
# cbar.set_label('Def (mm/yr)')
# cbar = fig.colorbar(im2, ax=axs[1,0], orientation='vertical', shrink=0.8)
# cbar.set_label('Reunwrapped (mm/yr)')
# cbar = fig.colorbar(im3, ax=axs[1,1], orientation='vertical', shrink=0.8)
# cbar.set_label('Def - LOS PM Corr (mm/yr)')
# cbar = fig.colorbar(im4, ax=axs[0,1], orientation='vertical', shrink=0.8)
# cbar.set_label('Reunw + LOS PM Corr (mm/yr)')
# axs[0,0].set_ylabel('Longitude (°E)')
# axs[1,1].set_xlabel('Latitude (°N)')
# fig.suptitle('Velocities and corrections, ITRF 2014, rel to {}, {} Spacing'.format(ref, post_lon_grid), y = 0.95)
# plt.subplots_adjust(wspace=0.4)
# plt.savefig('./outputs/{}/grids/{}_vels_PM_corr_{}_ITRF2014_{}_pm_ref.jpg'.format(insar_frame, insar_frame, post_lon_grid, ref), dpi=400, bbox_inches='tight')
# plt.show()

#%% convert imdates to good format
dates_rw = []
for date_num in imdates_rw:
        date_str = str(date_num)      
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        dates_rw.append(date_obj)
    
dates_def = []
for date_num in imdates_def:
        date_str = str(date_num)      
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        dates_def.append(date_obj)

#%% Read lat and lon from nc file

#lat = nc_dataset.variables['lat'][:]
#lon = nc_dataset.variables['lon'][:]
#cum = np.flip(nc_dataset.variables['cum'], axis=0)
#vel = np.flip(nc_dataset.variables['vel'][:,:], axis=0)

#%% Define the extent of the image using latitude and longitude values
lat_min_rw, lat_max_rw = lat_rw.min(), lat_rw.max()
lon_min_rw, lon_max_rw = lon_rw.min(), lon_rw.max()

lat_min_def, lat_max_def = lat_def.min(), lat_def.max()
lon_min_def, lon_max_def = lon_def.min(), lon_def.max()

#%% Find the indices of the nearest grid cell to the poi
lat_index_rw = np.abs(lat_rw - gnss_lat).argmin()
lon_index_rw = np.abs(lon_rw - gnss_lon).argmin()

lat_index_def = np.abs(lat_def - gnss_lat).argmin()
lon_index_def = np.abs(lon_def - gnss_lon).argmin()

#%%
# Extract the subset of data within the buffer
# cum_ts = np.flip(cum_rw[:, lat_index, lon_index], axis=0) # if using nc file
cum_ts_rw = cum_rw[:, lat_index_rw, lon_index_rw] # if using cum file
#%%
cum_ts_def = cum_def[:, lat_index_def, lon_index_def] # if using cum file

#%% better way
dfGPS['Dates'] = pd.to_datetime(dfGPS['Dates'], format='%Y') + pd.to_timedelta((dfGPS['Dates'] % 1) * 365, unit='D')

#%% Find indices to index InSAR data to desired date range
# Define the desired date range
start_date = datetime.strptime(start_date, "%Y%m%d")
end_date = datetime.strptime(end_date, "%Y%m%d")
#%%
# Find the indices of the 'dates' array that correspond to the desired date range
start_index_rw = next(idx for idx, t in enumerate(dates_rw) if t >= start_date)
end_index_rw = next(idx for idx, t in enumerate(dates_rw) if t <= end_date)
end_index_rw = next((idx for idx, date in enumerate(dates_rw) if date > end_date), len(dates_rw))
end_index_rw -= 1

start_index_def = next(idx for idx, t in enumerate(dates_def) if t >= start_date)
end_index_def = next(idx for idx, t in enumerate(dates_def) if t <= end_date)
end_index_def = next((idx for idx, date in enumerate(dates_def) if date > end_date), len(dates_def))
end_index_def -= 1


#Do the same for the GNSS data
# Find the indices of the 'dates' array that correspond to the desired date range
start_index_gnss = next(idx for idx, t in enumerate(dfGPS.Dates) if t >= start_date)
end_index_gnss = next(idx for idx, t in enumerate(dfGPS.Dates) if t <= end_date)
end_index_gnss = next((idx for idx, date in enumerate(dfGPS.Dates) if date > end_date), len(dfGPS.Dates))
end_index_gnss -= 1

#%% Extract subsets of datasets
#%% Convert InSAR dates to decimals for calculations
# Taken from https://github.com/sczesla/PyAstronomy/blob/master/src/pyasl/asl/decimalYear.py
dates_dec_rw = []
for d in dates_rw:
    year = d.year
    startOfThisYear = datetime(year=year, month=1, day=1)
    startOfNextYear = datetime(year=year+1, month=1, day=1)
    yearElapsed = (d) - (startOfThisYear)
    yearDuration = (startOfNextYear) - (startOfThisYear)
    fraction = yearElapsed/yearDuration
    date_dec = year + fraction
    dates_dec_rw.append(date_dec)
    
# Extract the subset of 'cum' data for the desired date range (InSAR reunw)
cum_subset_rw = cum_ts_rw[start_index_rw:end_index_rw+1]
dates_subset_rw = dates_rw[start_index_rw:end_index_rw+1]
dates_dec_subset_rw = dates_dec_rw[start_index_rw:end_index_rw+1]

#%% GNSS
gnss_v_subset = dfGPS.dU[start_index_gnss:end_index_gnss+1]
gnss_e_subset = dfGPS.dE[start_index_gnss:end_index_gnss+1]
gnss_n_subset = dfGPS.dN[start_index_gnss:end_index_gnss+1]
gnss_dates_subset = dfGPS.Dates[start_index_gnss:end_index_gnss+1]
gnss_dec_dates_subset = dfgnss['yyyy.yyyy'][start_index_gnss:end_index_gnss+1]

#%%
# Plot the subset of 'cum' data
plt.plot(dates_subset_rw, cum_subset_rw)

# Add labels and title to the plot
plt.xlabel('Time')
plt.ylabel('Cumulative')
plt.title('Cumulative Data')

#%% Convert InSAR dates to decimals for calculations
# Taken from https://github.com/sczesla/PyAstronomy/blob/master/src/pyasl/asl/decimalYear.py
dates_dec_def = []
for d in dates_def:
    year = d.year
    startOfThisYear = datetime(year=year, month=1, day=1)
    startOfNextYear = datetime(year=year+1, month=1, day=1)
    yearElapsed = (d) - (startOfThisYear)
    yearDuration = (startOfNextYear) - (startOfThisYear)
    fraction = yearElapsed/yearDuration
    date_dec = year + fraction
    dates_dec_def.append(date_dec)
    
# Extract the subset of 'cum' data for the desired date range (default)
cum_subset_def = cum_ts_def[start_index_def:end_index_def+1]
dates_subset_def = dates_def[start_index_def:end_index_def+1]
dates_dec_subset_def = dates_dec_def[start_index_def:end_index_def+1]

# Plot the subset of 'cum' data
plt.plot(dates_subset_def, cum_subset_def)

# Add labels and title to the plot
plt.xlabel('Time')
plt.ylabel('Cumulative')
plt.title('Cumulative Data')

#%%
## Calculate incidence angle at the GNSS site
LOSu = np.fromfile(LOSufile, dtype='float32').reshape((length_def, width_def))
inc_agl_deg = np.rad2deg(np.arccos(LOSu))
inc_agl = inc_agl_deg[lat_index_def, lon_index_def]
#%%
### Change to LOS
## Frame 078A_07049_131313
## Take inc and heading from LiCS Portal metadata.txt for frame of interest
inc=inc_agl*(np.pi/180)
head=-10.804014
az=(360+head)*(np.pi/180)
sininc=math.sin(inc)
cosinc=math.cos(inc)
sinaz=math.sin(az)
cosaz=math.cos(az)
GPS_dLOS = (((gnss_n_subset*sinaz)-(gnss_e_subset*cosaz))*sininc)+(gnss_v_subset*cosinc)

#%%

fig=plt.figure(figsize=(20,20))
ax = fig.add_subplot(111, polar=True)

ax1 = plt.subplot(4,1,1)
trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
ax1.text(0.0, 0.1, "a)", transform=ax1.transAxes + trans,fontsize='large', verticalalignment='top',     bbox=dict(facecolor='white', edgecolor='none', pad=3.0))
ax1.set_title("GNSS Vertical Displacement ({})".format(gnss_station), fontsize=16)
plt.plot(gnss_dates_subset, gnss_v_subset, color='blue', marker="o", label='N', linestyle='None', markersize=6, linewidth=0.5)

ax2 = plt.subplot(4,1,2, sharex=ax1)
trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
ax2.text(0.0, 0.1, "b)", transform=ax2.transAxes + trans,fontsize='large', verticalalignment='top',     bbox=dict(facecolor='white', edgecolor='none', pad=3.0))
### Plot the GPS in LOS check the equation 
plt.plot(gnss_dates_subset, GPS_dLOS, color='blue', marker="o", label='N', linestyle='None', markersize=6, linewidth=0.5)
ax2.set_title("GNSS LoS Displacement ({})".format(gnss_station), fontsize=16)
fig.add_subplot(111, frame_on=False)
plt.tick_params(labelcolor="none", bottom=False, left=False)


ax3 = plt.subplot(4,1,3, sharex=ax1)
trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
ax3.text(0.0, 0.1, "c)", transform=ax3.transAxes + trans,fontsize='large', verticalalignment='top',     bbox=dict(facecolor='white', edgecolor='none', pad=3.0))
plt.plot(dates_subset_rw, cum_subset_rw, color='blue', marker="o", label='N', linestyle='None', markersize=6, linewidth=0.5)
ax3.set_title("Reunwrapped ({}) InSAR LoS Displacement".format(insar_frame), fontsize=16)
#ax3.xaxis.set_visible(False)

ax4 = plt.subplot(4,1,4, sharex=ax1)
trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
ax4.text(0.0, 0.1, "d)", transform=ax4.transAxes + trans,fontsize='large', verticalalignment='top',     bbox=dict(facecolor='white', edgecolor='none', pad=3.0))
plt.plot(dates_subset_def, cum_subset_def, color='blue', marker="o", label='N', linestyle='None', markersize=6, linewidth=0.5)
ax4.set_title("Default ({}) InSAR LoS Displacement".format(insar_frame), fontsize=16)

plt.xlabel('Dates', fontsize=18,fontweight='bold') 
plt.ylabel('mm', fontsize=18, x= -5)   
fig.suptitle("Displacement time-series at point {}, {}".format(gnss_lon,gnss_lat), fontweight='bold', fontsize=18, y=0.99)

plt.tight_layout()
plt.savefig('./outputs/{}/gnss_insar_ts/{}_{}_{}_disp_ts_corr_inc.jpg'.format(insar_frame, insar_frame, gnss_station, ml_clip), format='jpg', dpi=400, bbox_inches='tight')
plt.show()

#%%
## Calculate linear fit for vertical GNSS using inverse theory
G = np.column_stack((date, np.ones(ndays)))

Q = np.concatenate((np.linalg.pinv(G.T.dot(np.diag(wE)).dot(G)),
                    np.linalg.pinv(G.T.dot(np.diag(wN)).dot(G)),
                    np.linalg.pinv(G.T.dot(np.diag(wU)).dot(G))), axis=1)

me = np.column_stack((Q[0:2, 0:2].dot(G.T.dot(np.diag(wE)).dot(E)),
                     Q[0:2, 2:4].dot(G.T.dot(np.diag(wN)).dot(N)),
                     Q[0:2, 4:6].dot(G.T.dot(np.diag(wU)).dot(U))))

me_plot= str(round(me[0,2], 2))
Q_plot = str(round(np.sqrt(Q[0,4]), 2))
me_line = me[0,2] * gnss_dec_dates_subset + me[1,2]
#%% Calculate MIDAS for vertical GNSS
# Ordinary Theil-Sen Estimator
# Have assumed one year, so dividing by 1 to get mm/yr 

# Robust way to deal with missing dates based upon Julian Day
forwardpairs = np.zeros((len(jday),3));

for n in range(0, jday.shape[0]):
     j = jday[n];
     d0 = np.where(jday==j)[0];
     d1 = np.where(jday==j+365)[0];
     if d0>=0 and d1>=0:
         forwardpairs[n, 0] = E[d1] - E[d0]
         forwardpairs[n, 1] = N[d1] - N[d0]
         forwardpairs[n, 2] = U[d1] - U[d0]
     else:
         forwardpairs[n,0:3] = np.NaN;

# % Paper says they run the algorithm forward and backwards as magnitude of
# % velocity should be same
backwardpairs = np.zeros((len(jday),3));

for n in range(jday.shape[0]-1, -1, -1):
     j = jday[n];
     d0 = np.where(jday==j)[0];
     d1 = np.where(jday==j-365)[0];
     if d0>=0 and d1>=0:
          backwardpairs[n, 0] = E[d0] - E[d1]
          backwardpairs[n, 1] = N[d0] - N[d1]
          backwardpairs[n, 2] = U[d0] - U[d1]
     else:
         backwardpairs[n,:] = np.NaN;

# % Combine Fore/Back Pairs together
pairs = np.concatenate((forwardpairs, backwardpairs), axis=0)

# Extract the valid values for each component
E1yr = pairs[np.isfinite(pairs[:, 0]), 0]
N1yr = pairs[np.isfinite(pairs[:, 1]), 1]
U1yr = pairs[np.isfinite(pairs[:, 2]), 2]

## # %% MIDAS Approach - Estimate dispersion and Trim Tails to re-estimate
# % median of absolute deviations (MAD).
MAD = np.median(abs(E1yr-np.median(E1yr)))
stdE = 1.4826*MAD;
MAD = np.median(abs(N1yr-np.median(N1yr)))
stdN = 1.4826*MAD;
MAD = np.median(abs(U1yr-np.median(U1yr)))
stdU = 1.4826*MAD;

# % Trim tails
E1yrtrim=E1yr[abs(E1yr-np.median(E1yr))<(2*stdE)];
N1yrtrim=N1yr[abs(N1yr-np.median(N1yr))<(2*stdN)];
U1yrtrim=U1yr[abs(U1yr-np.median(U1yr))<(2*stdU)];

# % Calculate Error in Trimmed Median
E1yrtrimerr = 1.4826*np.median(abs(E1yrtrim-np.median(E1yrtrim)));
N1yrtrimerr = 1.4826*np.median(abs(N1yrtrim-np.median(N1yrtrim)));
U1yrtrimerr = 1.4826*np.median(abs(U1yrtrim-np.median(U1yrtrim)));

# %% Uncertinity in the Velocity
# % assumes trimmed distribution is normal
# % standard error in the median = sqrt(pi/2) sigma(sqrt N)
# % N = Nactual/4 as using pairs and going backwards and forwards
NE = len(E1yrtrim)/4; NN = len(N1yrtrim)/4; NU = len(U1yrtrim)/4;
E1yrtrimstderr = np.sqrt(np.pi/2)*(E1yrtrimerr/np.sqrt(NE));
N1yrtrimstderr = np.sqrt(np.pi/2)*(N1yrtrimerr/np.sqrt(NN));
U1yrtrimstderr = np.sqrt(np.pi/2)*(U1yrtrimerr/np.sqrt(NU));

# % Multiple up by factor of three to account for data being autocorrelated
E1yrtrimstderr_midas = 3*E1yrtrimstderr;
N1yrtrimstderr_midas = 3*N1yrtrimstderr;
U1yrtrimstderr_midas = 3*U1yrtrimstderr;

#%%
# Reproject MIDAS vertical GNSS vel and err into LOS GNSS vertical
MIDAS_los = (((N1yr*sinaz)-(E1yr*cosaz))*sininc)+(U1yr*cosinc)
MIDAS_los_err = (((N1yrtrimstderr_midas*sinaz)-(E1yrtrimstderr_midas*cosaz))*sininc)+(U1yrtrimstderr_midas*cosinc)
#%%
## Calculate linear regression for LoS GNSS
slope, intercept, r_value, p_value, std_err = linregress(gnss_dec_dates_subset, GPS_dLOS)
slope_plot = format(slope, '.2f')
std_err_plot = format(std_err, '.2f')
regression_line = slope * gnss_dec_dates_subset + intercept

## Calculate linear regression for LoS InSAR
# Filter out NaN values
valid_indices = ~np.isnan(cum_subset_rw)
dates_dec_subset_rw = (np.array(dates_dec_subset_rw))[valid_indices]
cum_subset_rw = (np.array(cum_subset_rw))[valid_indices]
dates_subset_rw = (np.array(dates_subset_rw))[valid_indices]
slope_i, intercept_i, r_value_i, p_value_i, std_err_i = linregress(dates_dec_subset_rw, cum_subset_rw)
slope_plot_i = format(slope_i, '.2f')
std_err_plot_i = format(std_err_i, '.2f')
regression_line_i = slope_i * np.array(dates_dec_subset_rw) + intercept_i

## Calculate linear regression for LoS InSAR
valid_indices = ~np.isnan(cum_subset_def)
dates_dec_subset_def = (np.array(dates_dec_subset_def))[valid_indices]
cum_subset_def = (np.array(cum_subset_def))[valid_indices]
dates_subset_def = (np.array(dates_subset_def))[valid_indices]
slope_d, intercept_d, r_value_d, p_value_d, std_err_d = linregress(dates_dec_subset_def, cum_subset_def)
slope_plot_d = format(slope_d, '.2f')
std_err_plot_d = format(std_err_d, '.2f')
regression_line_d = slope_d * np.array(dates_dec_subset_def) + intercept_d

#%% Plot time-series with linear-fit
fig=plt.figure(figsize=(20,20))
ax = fig.add_subplot(111, polar=True)

ax1 = plt.subplot(4,1,1)
#trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
ax1.text(0.0, 0.1, "a)", transform=ax1.transAxes + trans,fontsize='large', verticalalignment='top',     bbox=dict(facecolor='white', edgecolor='none', pad=3.0))
ax1.set_title("GNSS Vertical Displacement ({})".format(gnss_station), fontsize=16)
plt.plot(gnss_dates_subset, gnss_v_subset, color='blue', marker="o", label='N', linestyle='None', markersize=6, linewidth=0.5)
plt.plot(gnss_dates_subset, (me[0,2] * gnss_dec_dates_subset + me[1,2]), 'g')
plt.text(gnss_dates_subset.iloc[150], -60, 'Linear fit: ' + me_plot + ' +/- ' + Q_plot + ' mm/yr', fontsize=16)

ax2 = plt.subplot(4,1,2, sharex=ax1)
#trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
ax2.text(0.0, 0.1, "b)", transform=ax2.transAxes + trans,fontsize='large', verticalalignment='top',     bbox=dict(facecolor='white', edgecolor='none', pad=3.0))
### Plot the GPS in LOS check the equation 
plt.plot(gnss_dates_subset, GPS_dLOS, color='blue', marker="o", label='N', linestyle='None', markersize=6, linewidth=0.5)
plt.plot(gnss_dates_subset, regression_line, 'g')
#plt.text(gnss_dates_subset.iloc[150], -60, 'Linear fit: ' + slope_plot + ' +/- ' + std_err_plot + ' mm/yr', fontsize=16)
ax2.set_title("GNSS LoS Displacement ({})".format(gnss_station), fontsize=16)


ax3 = plt.subplot(4,1,3, sharex=ax1)
#trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
ax3.text(0.0, 0.1, "c)", transform=ax3.transAxes + trans,fontsize='large', verticalalignment='top',     bbox=dict(facecolor='white', edgecolor='none', pad=3.0))
plt.plot(dates_subset_rw, cum_subset_rw, color='blue', marker="o", label='N', linestyle='None', markersize=6, linewidth=0.5)
plt.plot(dates_subset_rw, regression_line_i, 'g')
#plt.text(dfGPS.Dates.iloc[150], -60, 'Linear fit: ' + slope_plot_i + ' +/- ' + std_err_plot_i + ' mm/yr', fontsize=16)
ax3.set_title("Ascending ({}) InSAR LoS Displacement".format(insar_frame), fontsize=16)

ax4 = plt.subplot(4,1,4, sharex=ax1)
trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
ax4.text(0.0, 0.1, "d)", transform=ax4.transAxes + trans,fontsize='large', verticalalignment='top',     bbox=dict(facecolor='white', edgecolor='none', pad=3.0))
plt.plot(dates_subset_def, cum_subset_def, color='blue', marker="o", label='N', linestyle='None', markersize=6, linewidth=0.5)
plt.plot(dates_subset_def, regression_line_d, 'g')
##plt.text(dfGPS.Dates.iloc[150], -60, 'Linear fit: ' + slope_plot_d + ' +/- ' + std_err_plot_d + ' mm/yr', fontsize=16)
ax4.set_title("Default ({}) InSAR LoS Displacement".format(insar_frame), fontsize=16)

plt.xlabel('Dates', fontsize=18,fontweight='bold') 
plt.ylabel('mm', fontsize=18, x= -5)   
fig.suptitle("Displacement time-series at point {}, {}".format(gnss_lon,gnss_lat), fontweight='bold', fontsize=18, y=0.98)

plt.tight_layout()
plt.savefig('./outputs/{}/lin_fit/{}_{}_{}_disp_ts_lin_fit_corr_inc.jpg'.format(insar_frame, insar_frame, gnss_station, ml_clip), dpi=400, bbox_inches='tight')
plt.show()

#%% Start all time-series at 0 to compare
GPS_dLOS_first = GPS_dLOS.iloc[0]
GPS_dLOS_zero = GPS_dLOS - GPS_dLOS_first

dfGPS_dU_first = gnss_v_subset.iloc[0]
dfGPS_dU_zero = gnss_v_subset- dfGPS_dU_first

cum_first_rw = cum_subset_rw[0]
cum_zero_rw = cum_subset_rw - cum_first_rw

cum_first_def = cum_subset_def[0]
cum_zero_def = cum_subset_def - cum_first_def

#%% Calculate new linear fit for zeroed vels
## Zero linear fits
me_line_zero = me_line - me_line.iloc[0]
regression_line_zero = regression_line - regression_line.iloc[0]
regression_line_i_zero = regression_line_i - regression_line_i[0]
regression_line_d_zero = regression_line_d - regression_line_d[0]

#%% plot 'normalised' time-series
plt.figure(figsize=(12,8))
# plt.plot(gnss_dates_subset, dfGPS_dU_zero, label="Vertical GNSS ({})".format(gnss_station), marker="o", linestyle='None', color='blue', markersize=3)
# plt.plot(gnss_dates_subset, GPS_dLOS_zero, label="LoS GNSS ({})".format(gnss_station), marker="o", linestyle='None', color='cornflowerblue', markersize=3)
# plt.plot(dates_subset_rw, cum_zero_rw, label="Reunwrapped LoS InSAR ({})".format(insar_frame), marker="o", linestyle='None', color='green', markersize=3)
# plt.plot(dates_subset_def, cum_zero_def, label="Default LoS InSAR ({})".format(insar_frame), marker="o", linestyle='None', color='lightgreen', markersize=3)


# Plot zeroed linear fits
plt.plot(gnss_dates_subset, me_line_zero, label="Vertical GNSS Linear Fit ({})".format(gnss_station), color = 'blue')
plt.plot(gnss_dates_subset, regression_line_zero, label="LoS GNSS Linear Fit ({})".format(gnss_station), color = 'cornflowerblue')
plt.plot(dates_subset_rw, regression_line_i_zero, label="Reunwrapped LoS InSAR Linear Fit ({})".format(insar_frame), color = 'green')
plt.plot(dates_subset_def, regression_line_d_zero, label="Default LoS InSAR Linear Fit ({})".format(insar_frame), color = 'lightgreen')

# Plot linear velocity
# MMX1 900,[-20,-290]
# MXTM 300, [-20,-80]
# MXTX 5, [-55,-70]
# plt.text(gnss_dates_subset.iloc[300], -20, 'Vertical GNSS velocity: ' + me_plot + ' +/- ' + Q_plot + ' mm/yr', fontsize=12)
# plt.text(gnss_dates_subset.iloc[300], -40, 'MIDAS vertical GNSS fit: ' + f"{round(np.median(U1yr),2)} +/- {round(U1yrtrimstderr_midas,3)} mm/yr", fontsize=12)
# plt.text(gnss_dates_subset.iloc[300], -60, 'LoS GNSS velocity: ' + slope_plot + ' +/- ' + std_err_plot + ' mm/yr', fontsize=12)
# plt.text(gnss_dates_subset.iloc[300], -80, 'Reunw LoS InSAR velocity: ' + slope_plot_i + ' +/- ' + std_err_plot_i + ' mm/yr', fontsize=12)
# plt.text(gnss_dates_subset.iloc[300], -100, 'Default LoS InSAR velocity: ' + slope_plot_d + ' +/- ' + std_err_plot_d + ' mm/yr', fontsize=12)

# Place the text below the plot using plt.figtext()
plt.figtext(0.5, -0.1, 'Vertical GNSS velocity: ' + me_plot + ' +/- ' + Q_plot + ' mm/yr', fontsize=12, ha='center')
plt.figtext(0.5, -0.075, 'MIDAS vertical GNSS fit: ' + f"{round(np.median(U1yr), 2)} +/- {round(U1yrtrimstderr_midas, 3)} mm/yr", fontsize=12, ha='center')
plt.figtext(0.5, -0.05, 'LoS GNSS velocity: ' + slope_plot + ' +/- ' + std_err_plot + ' mm/yr', fontsize=12, ha='center')
plt.figtext(0.5, -0.025, 'MIDAS LoS GNSS fit: ' + f"{round(np.median(MIDAS_los), 2)} +/- {round(MIDAS_los_err, 3)} mm/yr", fontsize=12, ha='center')
plt.figtext(0.5, -0.0, 'Reunw LoS InSAR velocity: ' + slope_plot_i + ' +/- ' + std_err_plot_i + ' mm/yr', fontsize=12, ha='center')
plt.figtext(0.5, 0.025, 'Default LoS InSAR velocity: ' + slope_plot_d + ' +/- ' + std_err_plot_d + ' mm/yr', fontsize=12, ha='center')

plt.xlabel('Dates') 
plt.ylabel('mm') 
plt.title("Displacement time-series at point {}, {}".format(gnss_lon,gnss_lat), fontweight='bold')
plt.legend(loc='lower left', fontsize=11)

plt.savefig('./outputs/{}/zeroed_lin_fit/{}_{}_{}_disp_ts_lin_fit_zeroed_corr_inc.jpg'.format(insar_frame, insar_frame, gnss_station, ml_clip), dpi=400, bbox_inches='tight')

#%% Plot same profile but Lat v los velocity
titles_vels = ["Default", "Reunwrapped", "Default Filtered"]
vel_list = vel_rw, vel_def, vel_def_filt

# Store arrays in a dictionary
vel_dict = dict(zip(titles_vels, vel_list))
#%%
# Define start and end points of profile (lon, lat)
start_point = (-99.1, 19.85)
end_point = (-98.94, 19.07)

# Create an array of indices for the file
plt.figure(figsize=(6,10))

# Loop through arrays in the dictionary and plot a profile for each
lons, lats = np.linspace(start_point[0], end_point[0], np.min(vel_rw.shape)), np.linspace(start_point[1], end_point[1], np.min(vel_rw.shape))

lat_indx_rw =[]
lat_indx_def =[]
lon_indx_rw = []
lon_indx_def = []

for lat in lats:
    lat_indx_rw.append(np.abs(lat_rw - lat).argmin())
    lat_indx_def.append(np.abs(lat_def - lat).argmin())
       
for lon in lons:
    lon_indx_rw.append(np.abs(lon_rw - lon).argmin())
    lon_indx_def.append(np.abs(lon_def - lon).argmin())

values_rw = vel_rw[lat_indx_rw, lon_indx_rw]
values_def = vel_def[lat_indx_def, lon_indx_def]
values_def_filt = vel_def_filt_arr[lat_indx_def, lon_indx_def]

# Plot the profile
plt.plot(values_rw, lats, label="Reunwrapped")
plt.plot(values_def, lats, label="Default")
plt.plot(values_def_filt, lats, label="Default Filtered")
    
# Add legend and axis labels
plt.legend(loc='upper left', fontsize=12)
plt.ylabel('Latitude (deg)', fontsize=13)
plt.xlabel('LoS Velocity (mm/yr)', fontsize=13)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.title('{} LoS velocity profiles'.format(insar_frame))

# Set y-axis properties to plot on the right side
ax = plt.gca()
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()

plt.savefig('./outputs/{}/maps/{}_{}_rw_lat_profile.jpg'.format(insar_frame, insar_frame, ml_clip), dpi=400, bbox_inches='tight')

#%% Plot horizontal profile
start_point_horiz = (-99.2, 19.4)
end_point_horiz = (-98.87, 19.47)

# Create an array of indices for the file
plt.figure(figsize=(10,6))

# Make arrays for lon and lat 
lons, lats = np.linspace(start_point_horiz[0], end_point_horiz[0], np.min(vel_rw.shape)), np.linspace(start_point_horiz[1], end_point_horiz[1], np.min(vel_rw.shape))

lat_indx_rw =[]
lat_indx_def =[]
lon_indx_rw = []
lon_indx_def = []

for lat in lats:
    lat_indx_rw.append(np.abs(lat_rw - lat).argmin())
    lat_indx_def.append(np.abs(lat_def - lat).argmin())
       
for lon in lons:
    lon_indx_rw.append(np.abs(lon_rw - lon).argmin())
    lon_indx_def.append(np.abs(lon_def - lon).argmin())

## do tomorrow ##
values_rw = vel_rw[lat_indx_rw, lon_indx_rw]
values_def = vel_def[lat_indx_def, lon_indx_def]
values_def_filt = vel_def_filt_arr[lat_indx_def, lon_indx_def]

    #%%

# Plot the profile
plt.plot(lons, values_rw, label="Reunwrapped")
plt.plot(lons, values_def, label="Default")
plt.plot(lons, values_def_filt, label="Default Filtered")
    
# Add legend and axis labels
plt.legend(loc='lower left', fontsize=10)
plt.xlabel('Longitude (deg)', fontsize=13)
plt.ylabel('LoS Velocity (mm/yr)', fontsize=13)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.title('{} LoS velocity profiles'.format(insar_frame))

# Set y-axis properties to plot on the right side
ax = plt.gca()
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()

plt.savefig('./outputs/{}/maps/{}_{}_rw_lon_profile.jpg'.format(insar_frame, insar_frame, ml_clip), dpi=400, bbox_inches='tight')
plt.show()

#%%
# Plot the 'vel' variable with latitude and longitude on the axes
plt.imshow(vel_rw, extent=[lon_min_rw, lon_max_rw, lat_min_rw, lat_max_rw], cmap = cm.vik, vmin=-200, vmax=200)

# Add a marker for the specific point (gnss_lat, gnss_lon)
plt.plot(gnss_lon, gnss_lat, 'rx', markersize=8)

# Add profile line
plt.plot([end_point[0], start_point[0]], [end_point[1], start_point[1]], color='red')
plt.plot([end_point_horiz[0], start_point_horiz[0]], [end_point_horiz[1], start_point_horiz[1]], color='red')

# Add labels and title to the plot
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('{} reunwr {} + {}'.format(insar_frame, ml_clip, gnss_station))
colorbar = plt.colorbar()
colorbar.set_label('LoS Velocity (mm/yr)')
plt.savefig('./outputs/{}/maps/{}_{}_{}.reunw_insar_los_map.jpg'.format(insar_frame, insar_frame, gnss_station, ml_clip), dpi=400, bbox_inches='tight')
plt.show()

#%%
# Add a marker for the specific point (gnss_lat, gnss_lon)
plt.imshow(vel_def, extent=[lon_min_def, lon_max_def, lat_min_def, lat_max_def], cmap = cm.vik, vmin=-200, vmax=200)
plt.plot(gnss_lon, gnss_lat, 'rx', markersize=8)

plt.plot([end_point[0], start_point[0]], [end_point[1], start_point[1]], color='red')
plt.plot([end_point_horiz[0], start_point_horiz[0]], [end_point_horiz[1], start_point_horiz[1]], color='red')

# Add labels and title to the plot
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('{} {} + {}'.format(insar_frame, ml_clip, gnss_station))
colorbar = plt.colorbar()
colorbar.set_label('LoS Velocity (mm/yr)')
plt.savefig('./outputs/{}/maps/{}_{}_{}.insar_los_map.jpg'.format(insar_frame, insar_frame, gnss_station, ml_clip), dpi=400, bbox_inches='tight')
plt.show()