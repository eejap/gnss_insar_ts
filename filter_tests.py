#!/usr/bin/python
## P.Espin 
### PLOT time series and change the file of GPS to LOS
### Jess Payne update 10/07/23 (filter-test script):
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
data_directory = '/nfs/a285/homes/eejap/filter_tests'
reunwr_directory= '/nfs/a285/homes/eejap/reunwrap_tests'
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

h5_file_05k = data_directory + '/' + insar_frame + '/' + insar_frame + '_0_5km_cum_filt.h5'
h5_file_2k = data_directory + '/' + insar_frame + '/' + insar_frame + '_2km_cum_filt.h5'
h5_file_16k = data_directory + '/' + insar_frame + '/' + insar_frame + '_16km_cum_filt.h5'

h5_file_8d = data_directory + '/' + insar_frame + '/' + insar_frame + '_8d_cum_filt.h5'
h5_file_30d = data_directory + '/' + insar_frame + '/' + insar_frame + '_30d_cum_filt.h5'
h5_file_128d = data_directory + '/' + insar_frame + '/' + insar_frame + '_128d_cum_filt.h5'

par_file_def = reunwr_directory + '/' + insar_frame + '/' + insar_frame + '_' + ml_clip +'.dem_par'
LOSufile = reunwr_directory + '/' + insar_frame + '/' + insar_frame + '_' + ml_clip +'.U.geo'
LOSefile = reunwr_directory + '/' + insar_frame + '/' + insar_frame + '_' + ml_clip +'.E.geo'
LOSnfile = reunwr_directory + '/' + insar_frame + '/' + insar_frame + '_' + ml_clip +'.N.geo'

vel_05k = data_directory + '/' + insar_frame + '/' + insar_frame + '_0_5km.vel.filt.geo.tif'
vel_2k = data_directory + '/' + insar_frame + '/' + insar_frame + '_2km.vel.filt.geo.tif'
vel_16k = data_directory + '/' + insar_frame + '/' + insar_frame + '_16km.vel.filt.geo.tif'

vel_8d = data_directory + '/' + insar_frame + '/' + insar_frame + '_8d.vel.filt.geo.tif'
vel_30d = data_directory + '/' + insar_frame + '/' + insar_frame + '_30d.vel.filt.geo.tif'
vel_128d = data_directory + '/' + insar_frame + '/' + insar_frame + '_128d.vel.filt.geo.tif'

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

#%% make vel_tifs a tif not string
vel_tif_05k = imageio.imread(vel_05k)
#vel_tif_2k = imageio.imread(vel_2k)
vel_tif_16k = imageio.imread(vel_16k)
#vel_tif_128k = imageio.imread(vel_128k)


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
with h5py.File(h5_file_05k, 'r') as file:
    imdates_05k = file['imdates']
    imdates_05k = imdates_05k[:]  
    vel_05k = file['vel']
    vel_05k = vel_05k[:]
    cum_05k = file['cum']
    cum_05k = cum_05k[:]
    
with h5py.File(h5_file_16k, 'r') as file:
    imdates_16k = file['imdates']
    imdates_16k = imdates_16k[:]  
    vel_16k = file['vel']
    vel_16k = vel_16k[:]
    cum_16k = file['cum']
    cum_16k= cum_16k[:]

#%% complete using h5 file
# read array dimensions from par file
width = int(lib.get_par(par_file_def,'width'))
length = int(lib.get_par(par_file_def,'nlines'))

# get corner positions
corner_lat = float(lib.get_par(par_file_def,'corner_lat'))
corner_lon = float(lib.get_par(par_file_def,'corner_lon'))

# get post spacing (distance between velocity measurements)
post_lat = float(lib.get_par(par_file_def,'post_lat'))
post_lon = float(lib.get_par(par_file_def,'post_lon'))

# calculate grid spacings
lat = corner_lat + post_lat*np.arange(1,length+1) - post_lat/2
lon = corner_lon + post_lon*np.arange(1,width+1) - post_lon/2

#%% convert imdates to good format. All should have the same dates so use only 0.5km filter imdates.
dates = []
for date_num in imdates_05k:
        date_str = str(date_num)      
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        dates.append(date_obj)

#%% Define the extent of the image using latitude and longitude values
lat_min, lat_max = lat.min(), lat.max()
lon_min, lon_max = lon.min(), lon.max()

#%% Find the indices of the nearest grid cell to the poi
lat_index = np.abs(lat - gnss_lat).argmin()
lon_index = np.abs(lon - gnss_lon).argmin()

#%%
# Extract the subset of data within the buffer
# cum_ts = np.flip(cum_rw[:, lat_index, lon_index], axis=0) # if using nc file
cum_ts_05k= cum_05k[:, lat_index, lon_index] # if using cum file
cum_ts_16k= cum_16k[:, lat_index, lon_index] # if using cum file

#%% better way
dfGPS['Dates'] = pd.to_datetime(dfGPS['Dates'], format='%Y') + pd.to_timedelta((dfGPS['Dates'] % 1) * 365, unit='D')

#%% Find indices to index InSAR data to desired date range
# Define the desired date range
start_date = datetime.strptime(start_date, "%Y%m%d")
end_date = datetime.strptime(end_date, "%Y%m%d")
#%%
# Find the indices of the 'dates' array that correspond to the desired date range
start_index = next(idx for idx, t in enumerate(dates) if t >= start_date)
end_index = next(idx for idx, t in enumerate(dates) if t <= end_date)
end_index = next((idx for idx, date in enumerate(dates) if date > end_date), len(dates))
end_index -= 1

#Do the same for the GNSS data
# Find the indices of the 'dates' array that correspond to the desired date range
start_index_gnss = next(idx for idx, t in enumerate(dfGPS.Dates) if t >= start_date)
end_index_gnss = next(idx for idx, t in enumerate(dfGPS.Dates) if t <= end_date)
end_index_gnss = next((idx for idx, date in enumerate(dfGPS.Dates) if date > end_date), len(dfGPS.Dates))
end_index_gnss -= 1

#%% Convert InSAR dates to decimals for calculations
# Taken from https://github.com/sczesla/PyAstronomy/blob/master/src/pyasl/asl/decimalYear.py
dates_dec = []
for d in dates:
    year = d.year
    startOfThisYear = datetime(year=year, month=1, day=1)
    startOfNextYear = datetime(year=year+1, month=1, day=1)
    yearElapsed = (d) - (startOfThisYear)
    yearDuration = (startOfNextYear) - (startOfThisYear)
    fraction = yearElapsed/yearDuration
    date_dec = year + fraction
    dates_dec.append(date_dec)
    
# Extract the subset of 'cum' data for the desired date range (InSAR reunw), again using 0.5km dates only should be fine
cum_subset_05k = cum_ts_05k[start_index:end_index+1]
cum_subset_16k = cum_ts_16k[start_index:end_index+1]
dates_subset = dates[start_index:end_index+1]
dates_dec_subset = dates_dec[start_index:end_index+1]

#%% GNSS
gnss_v_subset = dfGPS.dU[start_index_gnss:end_index_gnss+1]
gnss_e_subset = dfGPS.dE[start_index_gnss:end_index_gnss+1]
gnss_n_subset = dfGPS.dN[start_index_gnss:end_index_gnss+1]
gnss_dates_subset = dfGPS.Dates[start_index_gnss:end_index_gnss+1]
gnss_dec_dates_subset = dfgnss['yyyy.yyyy'][start_index_gnss:end_index_gnss+1]

#%%
# Plot the subset of 'cum' data
plt.plot(dates_subset, cum_subset_05k)
plt.plot(dates_subset, cum_subset_16k)

# Add labels and title to the plot
plt.xlabel('Time')
plt.ylabel('Cumulative')
plt.title('Cumulative Data')

#%%
## Calculate incidence angle at the GNSS site
LOSu = np.fromfile(LOSufile, dtype='float32').reshape((length, width))
inc_agl_deg = np.rad2deg(np.arccos(LOSu))
inc_agl = inc_agl_deg[lat_index, lon_index]
#%%
### Change to LOS
## Frame 078A_07049_131313
## Take inc and heading from LiCS Portal metadata.txt for frame of interest
inc=inc_agl*(np.pi/180)
#head=-10.804014 # 078A
#head = -169.14888 # 041D
head = -169.11923 # 143D
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
plt.plot(dates_subset, cum_subset_05k, color='blue', marker="o", label='N', linestyle='None', markersize=6, linewidth=0.5)
ax3.set_title("0.5km ({}) InSAR LoS Displacement".format(insar_frame), fontsize=16)
#ax3.xaxis.set_visible(False)

ax4 = plt.subplot(4,1,4, sharex=ax1)
trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
ax4.text(0.0, 0.1, "d)", transform=ax4.transAxes + trans,fontsize='large', verticalalignment='top',     bbox=dict(facecolor='white', edgecolor='none', pad=3.0))
plt.plot(dates_subset, cum_subset_16k, color='blue', marker="o", label='N', linestyle='None', markersize=6, linewidth=0.5)
ax4.set_title("16km ({}) InSAR LoS Displacement".format(insar_frame), fontsize=16)

plt.xlabel('Dates', fontsize=18,fontweight='bold') 
plt.ylabel('mm', fontsize=18, x= -5)   
fig.suptitle("Displacement time-series at point {}, {}".format(gnss_lon,gnss_lat), fontweight='bold', fontsize=18, y=0.99)

plt.tight_layout()
#plt.savefig('./outputs/{}/gnss_insar_ts/{}_{}_{}_disp_ts_corr_inc.jpg'.format(insar_frame, insar_frame, gnss_station, ml_clip), format='jpg', dpi=400, bbox_inches='tight')
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
valid_indices = ~np.isnan(cum_subset_05k)
dates_dec_subset = (np.array(dates_dec_subset))[valid_indices]
cum_subset_05k = (np.array(cum_subset_05k))[valid_indices]
dates_subset = (np.array(dates_subset))[valid_indices]
slope_i, intercept_i, r_value_i, p_value_i, std_err_i = linregress(dates_dec_subset, cum_subset_05k)
slope_plot_i = format(slope_i, '.2f')
std_err_plot_i = format(std_err_i, '.2f')
regression_line_i = slope_i * np.array(dates_dec_subset) + intercept_i

## Calculate linear regression for LoS InSAR
valid_indices = ~np.isnan(cum_subset_16k)
dates_dec_subset = (np.array(dates_dec_subset))[valid_indices]
cum_subset_16k = (np.array(cum_subset_16k))[valid_indices]
dates_subset = (np.array(dates_subset))[valid_indices]
slope_d, intercept_d, r_value_d, p_value_d, std_err_d = linregress(dates_dec_subset, cum_subset_16k)
slope_plot_d = format(slope_d, '.2f')
std_err_plot_d = format(std_err_d, '.2f')
regression_line_d = slope_d * np.array(dates_dec_subset) + intercept_d

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
plt.plot(dates_subset, cum_subset_05k, color='blue', marker="o", label='N', linestyle='None', markersize=6, linewidth=0.5)
plt.plot(dates_subset, regression_line_i, 'g')
#plt.text(dfGPS.Dates.iloc[150], -60, 'Linear fit: ' + slope_plot_i + ' +/- ' + std_err_plot_i + ' mm/yr', fontsize=16)
ax3.set_title("Ascending ({}) InSAR LoS Displacement".format(insar_frame), fontsize=16)

ax4 = plt.subplot(4,1,4, sharex=ax1)
trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
ax4.text(0.0, 0.1, "d)", transform=ax4.transAxes + trans,fontsize='large', verticalalignment='top',     bbox=dict(facecolor='white', edgecolor='none', pad=3.0))
plt.plot(dates_subset, cum_subset_16k, color='blue', marker="o", label='N', linestyle='None', markersize=6, linewidth=0.5)
plt.plot(dates_subset, regression_line_d, 'g')
##plt.text(dfGPS.Dates.iloc[150], -60, 'Linear fit: ' + slope_plot_d + ' +/- ' + std_err_plot_d + ' mm/yr', fontsize=16)
ax4.set_title("Default ({}) InSAR LoS Displacement".format(insar_frame), fontsize=16)

plt.xlabel('Dates', fontsize=18,fontweight='bold') 
plt.ylabel('mm', fontsize=18, x= -5)   
fig.suptitle("Displacement time-series at point {}, {}".format(gnss_lon,gnss_lat), fontweight='bold', fontsize=18, y=0.98)

plt.tight_layout()
#plt.savefig('./outputs/{}/lin_fit/{}_{}_{}_disp_ts_lin_fit_corr_inc.jpg'.format(insar_frame, insar_frame, gnss_station, ml_clip), dpi=400, bbox_inches='tight')
plt.show()

#%% Start all time-series at 0 to compare
GPS_dLOS_first = GPS_dLOS.iloc[0]
GPS_dLOS_zero = GPS_dLOS - GPS_dLOS_first

dfGPS_dU_first = gnss_v_subset.iloc[0]
dfGPS_dU_zero = gnss_v_subset- dfGPS_dU_first

cum_first_05k = cum_subset_05k[0]
cum_zero_05k = cum_subset_05k - cum_first_05k

cum_first_16k = cum_subset_16k[0]
cum_zero_16k = cum_subset_16k - cum_first_16k

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
plt.plot(dates_subset, regression_line_i_zero, label="0.5km LoS InSAR Linear Fit ({})".format(insar_frame), color = 'green')
plt.plot(dates_subset, regression_line_d_zero, label="16km LoS InSAR Linear Fit ({})".format(insar_frame), color = 'lightgreen')

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
plt.figtext(0.5, -0.0, '0.5 km LoS InSAR velocity: ' + slope_plot_i + ' +/- ' + std_err_plot_i + ' mm/yr', fontsize=12, ha='center')
plt.figtext(0.5, 0.025, '16 km LoS InSAR velocity: ' + slope_plot_d + ' +/- ' + std_err_plot_d + ' mm/yr', fontsize=12, ha='center')

plt.xlabel('Dates') 
plt.ylabel('mm') 
plt.title("Displacement time-series at point {}, {}".format(gnss_lon,gnss_lat), fontweight='bold')
plt.legend(loc='lower left', fontsize=11)

#plt.savefig('./outputs/{}/zeroed_lin_fit/{}_{}_{}_disp_ts_lin_fit_zeroed_corr_inc.jpg'.format(insar_frame, insar_frame, gnss_station, ml_clip), dpi=400, bbox_inches='tight')

#%% Plot same profile but Lat v los velocity
titles_vels = ["0.5km", "16km"]
vel_list = vel_05k, vel_16k

# Store arrays in a dictionary
vel_dict = dict(zip(titles_vels, vel_list))
#%%
# Define start and end points of profile (lon, lat)
start_point = (-99.1, 19.85)
end_point = (-98.94, 19.07)

# Create an array of indices for the file
plt.figure(figsize=(6,10))

# Loop through arrays in the dictionary and plot a profile for each
lons, lats = np.linspace(start_point[0], end_point[0], np.min(vel_05k.shape)), np.linspace(start_point[1], end_point[1], np.min(vel_05k.shape))

lat_indx =[]
lon_indx = []

for lat2 in lats:
    lat_indx.append(np.abs(lat - lat2).argmin())
       
for lon2 in lons:
    lon_indx.append(np.abs(lon - lon2).argmin())

values_05k = vel_05k[lat_indx, lon_indx]
values_16k = vel_16k[lat_indx, lon_indx]

# Plot the profile
plt.plot(values_05k, lats, label="0.5 km")
plt.plot(values_16k, lats, label="16 km")
    
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

#plt.savefig('./outputs/{}/maps/{}_{}_rw_lat_profile.jpg'.format(insar_frame, insar_frame, ml_clip), dpi=400, bbox_inches='tight')

#%% Plot horizontal profile
start_point_horiz = (-99.2, 19.4)
end_point_horiz = (-98.87, 19.47)

# Create an array of indices for the file
plt.figure(figsize=(10,6))

# Make arrays for lon and lat 
lons, lats = np.linspace(start_point_horiz[0], end_point_horiz[0], np.min(vel_05k.shape)), np.linspace(start_point_horiz[1], end_point_horiz[1], np.min(vel_05k.shape))

lat_indx =[]
lon_indx = []

for lat2 in lats:
    lat_indx.append(np.abs(lat - lat2).argmin())
       
for lon2 in lons:
    lon_indx.append(np.abs(lon - lon2).argmin())

values_05k = vel_05k[lat_indx, lon_indx]
values_16k = vel_16k[lat_indx, lon_indx]

    #%%

# Plot the profile
plt.plot(lons, values_05k, label="0.5 km")
plt.plot(lons, values_16k, label="16 km")
    
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

#plt.savefig('./outputs/{}/maps/{}_{}_rw_lon_profile.jpg'.format(insar_frame, insar_frame, ml_clip), dpi=400, bbox_inches='tight')
plt.show()

#%%
# Plot the 'vel' variable with latitude and longitude on the axes
plt.imshow(vel_05k, extent=[lon_min, lon_max, lat_min, lat_max], cmap = cm.vik, vmin=-200, vmax=200)

# Add a marker for the specific point (gnss_lat, gnss_lon)
plt.plot(gnss_lon, gnss_lat, 'rx', markersize=8)

# Add profile line
plt.plot([end_point[0], start_point[0]], [end_point[1], start_point[1]], color='red')
plt.plot([end_point_horiz[0], start_point_horiz[0]], [end_point_horiz[1], start_point_horiz[1]], color='red')

# Add labels and title to the plot
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('{} 0.5 km filter {} + {}'.format(insar_frame, ml_clip, gnss_station))
colorbar = plt.colorbar()
colorbar.set_label('LoS Velocity (mm/yr)')
#plt.savefig('./outputs/{}/maps/{}_{}_{}.reunw_insar_los_map.jpg'.format(insar_frame, insar_frame, gnss_station, ml_clip), dpi=400, bbox_inches='tight')
plt.show()

#%%
# Add a marker for the specific point (gnss_lat, gnss_lon)
plt.imshow(vel_16k, extent=[lon_min, lon_max, lat_min, lat_max], cmap = cm.vik, vmin=-200, vmax=200)
plt.plot(gnss_lon, gnss_lat, 'rx', markersize=8)

plt.plot([end_point[0], start_point[0]], [end_point[1], start_point[1]], color='red')
plt.plot([end_point_horiz[0], start_point_horiz[0]], [end_point_horiz[1], start_point_horiz[1]], color='red')

# Add labels and title to the plot
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('{} 16 km filter {} + {}'.format(insar_frame, ml_clip, gnss_station))
colorbar = plt.colorbar()
colorbar.set_label('LoS Velocity (mm/yr)')
#plt.savefig('./outputs/{}/maps/{}_{}_{}.insar_los_map.jpg'.format(insar_frame, insar_frame, gnss_station, ml_clip), dpi=400, bbox_inches='tight')
plt.show()