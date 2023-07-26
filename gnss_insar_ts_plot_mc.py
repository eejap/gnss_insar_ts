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
insar_frame = "078A_07049_131313"
ml_clip = "ml1_clip" # ml10 or ml1_clip

filename = directorio + '/data/gnss/' + gnss_station + '.NA.tenv3'
gnss_station_info = directorio + '/data/gnss/gnss_info.txt'
formatSpec = '%4C%8s%10f%6f%5f%2f%7f%7f%10f%10f%10f%6f%10f%8f%9f%9f%9f%10f%10f%f%[^\n\r]'

par_file_rw = data_directory + '/' + insar_frame + '/' + insar_frame + '_gucmT_' + ml_clip +'.dem_par' # use same par file and E, N, U files for all reunwrapped results
par_file_def = data_directory + '/' + insar_frame + '/' + insar_frame + '_' + ml_clip +'.dem_par'
LOSufile = data_directory + '/' + insar_frame + '/' + insar_frame + '_' + ml_clip +'.U.geo'
LOSefile = data_directory + '/' + insar_frame + '/' + insar_frame + '_' + ml_clip +'.E.geo'
LOSnfile = data_directory + '/' + insar_frame + '/' + insar_frame + '_' + ml_clip +'.N.geo'

#%%
dfgnss = pd.read_csv(filename, delimiter=r"\s+")
dfgnss_info = pd.read_csv(gnss_station_info, delimiter=r"\s+")
dfgnss_info.set_index('name', inplace=True)

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

#%% Import reunwrapped cum files
ruw_list = ['default', 'gaussian', 'lowpass', 'goldstein_cascade']
tif_end = '.cum.h5'
file_dir = '/nfs/a285/homes/eejap/reunwrap_tests/{}/reunw_cum/'.format(insar_frame) # all data in this dir is ml1!!

#%%
#  Create list of cum.h5 names
cum_files = []
for i in range(len(ruw_list)):
    tif = file_dir + ruw_list[i] + tif_end
    cum_files.append(tif)
#%%
# Import cum files and save them in list called cum_files
imdates_list =[]
vel_list = []
cum_list = []

for filex in cum_files:
    with h5py.File(filex, 'r') as file:
        imdates = file['imdates']
        imdates = imdates[:] 
        imdates_list.append(imdates)
        vel = file['vel']
        vel = vel[:]
        vel_list.append(vel)
        cum = file['cum']
        cum = cum[:]
        cum_list.append(cum)
#%% Draw profiles through velocities
titles_vels = ["No reunwrapping", "Gaussian", "Lowpass", "Goldstein + Cascade"]


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

#%% convert imdates to good format, store in dictionary
imdates_dict = dict(zip(titles_vels, imdates_list))

dates_dict = {}
for i, name in enumerate(imdates_dict):
    imdates = imdates_dict[name]
    dates_temp = []
    for dates in imdates:
        date_str = str(dates) 
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        dates_temp.append(date_obj)
    dates_dict[name] = dates_temp # dictionary holds better formatted dates for bothversions of the velocities
    

#%% Define the extent of the image using latitude and longitude values. Use same extents for all rw images.
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
# Extract the time-series at the point closes to the GNSS
cum_dict = dict(zip(titles_vels, cum_list))

cum_ts_dict = {}
for i, name in enumerate(cum_dict):
    cums = cum_dict[name]
    if name == 'No reunwrapping':
        cum_ts = cums[:, lat_index_def, lon_index_def] # if using cum file
    else:
        cum_ts = cums[:, lat_index_rw, lon_index_rw] # if using cum file
    cum_ts_dict[name] = cum_ts # dictionary holds time-series displacement data for point at closes point to GNSS location
    

#%% better way
dfGPS['Dates'] = pd.to_datetime(dfGPS['Dates'], format='%Y') + pd.to_timedelta((dfGPS['Dates'] % 1) * 365, unit='D')

#%% Find indices to index InSAR data to desired date range
# Define the desired date range
start_date = datetime.strptime(start_date, "%Y%m%d")
end_date = datetime.strptime(end_date, "%Y%m%d")
#%%
# Find the indices of the 'dates' array that correspond to the desired date range
# extract first reunwrapped dates
dates_rw = dates_dict['Goldstein + Cascade']
start_index_rw = next(idx for idx, t in enumerate(dates_rw) if t >= start_date)
end_index_rw = next(idx for idx, t in enumerate(dates_rw) if t <= end_date)
end_index_rw = next((idx for idx, date in enumerate(dates_rw) if date > end_date), len(dates_rw))
end_index_rw -= 1

# extract default imdates
dates_def = dates_dict['No reunwrapping']
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

#%% Convert InSAR dates to decimals for calculations
# Taken from https://github.com/sczesla/PyAstronomy/blob/master/src/pyasl/asl/decimalYear.py

dates_dec_dict = {}
for i, name in enumerate(dates_dict):
    dates = dates_dict[name]
    dates_temp = []
    for d in dates:
        year = d.year
        startOfThisYear = datetime(year=year, month=1, day=1)
        startOfNextYear = datetime(year=year+1, month=1, day=1)
        yearElapsed = (d) - (startOfThisYear)
        yearDuration = (startOfNextYear) - (startOfThisYear)
        fraction = yearElapsed/yearDuration
        date_dec = year + fraction
        dates_temp.append(date_dec)
    dates_dec_dict[name] = dates_temp
    
  #%%  
# Extract the subset of 'cum' data for the desired date range (InSAR reunw)

cum_subset_ts_dict = {}
dates_subset_dict = {}
dates_dec_subset_dict = {}
for i, name in enumerate(cum_ts_dict):
    cums = cum_ts_dict[name]
    dates_dec = dates_dec_dict[name]
    dates = dates_dict[name]
    if name == 'No reunwrapping':
        cum_subset = cums[start_index_def:end_index_def+1]
        dates_subset = dates[start_index_def:end_index_def+1]
        dates_dec_subset = dates_dec[start_index_def:end_index_def+1]
    else:         
        cum_subset = cums[start_index_rw:end_index_rw+1]
        dates_subset = dates_rw[start_index_rw:end_index_rw+1]
        dates_dec_subset = dates_dec[start_index_rw:end_index_rw+1]
    cum_subset_ts_dict[name] = cum_subset
    dates_subset_dict[name] = dates_subset
    dates_dec_subset_dict[name] = dates_dec_subset
    

#%% GNSS
gnss_v_subset = dfGPS.dU[start_index_gnss:end_index_gnss+1]
gnss_e_subset = dfGPS.dE[start_index_gnss:end_index_gnss+1]
gnss_n_subset = dfGPS.dN[start_index_gnss:end_index_gnss+1]
gnss_dates_subset = dfGPS.Dates[start_index_gnss:end_index_gnss+1]
gnss_dec_dates_subset = dfgnss['yyyy.yyyy'][start_index_gnss:end_index_gnss+1]

#%%
# Plot the subset of 'cum' data
for i, name in enumerate(dates_subset_dict):
    dates_plot = dates_subset_dict[name]
    cum_plot = cum_subset_ts_dict[name]
    plt.plot(dates_plot, cum_plot)

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
## Take heading from LiCS Portal metadata.txt for frame of interest
inc=inc_agl*(np.pi/180)
#head=-10.804014 # 078A, MC
head=-169.14888 # 041D, MC
#head=-169.11923 # 143D, MC
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
plt.plot(dates_subset_dict['Goldstein + Cascade'], cum_subset_ts_dict['Goldstein + Cascade'], color='blue', marker="o", label='N', linestyle='None', markersize=6, linewidth=0.5)
ax3.set_title("Goldstein + Cascade ({}) InSAR LoS Displacement".format(insar_frame), fontsize=16)
#ax3.xaxis.set_visible(False)

ax4 = plt.subplot(4,1,4, sharex=ax1)
trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
ax4.text(0.0, 0.1, "d)", transform=ax4.transAxes + trans,fontsize='large', verticalalignment='top',     bbox=dict(facecolor='white', edgecolor='none', pad=3.0))
plt.plot(dates_subset_dict['No reunwrapping'], cum_subset_ts_dict['No reunwrapping'], color='blue', marker="o", label='N', linestyle='None', markersize=6, linewidth=0.5)
ax4.set_title("No reunwrapping ({}) InSAR LoS Displacement".format(insar_frame), fontsize=16)

plt.xlabel('Dates', fontsize=18,fontweight='bold') 
plt.ylabel('mm', fontsize=18, x= -5)   
fig.suptitle("Displacement time-series at point {}, {}".format(gnss_lon,gnss_lat), fontweight='bold', fontsize=18, y=0.99)

plt.tight_layout()
#plt.savefig('/nfs/a285/homes/eejap/reunwrap_tests/078A_07049_131313/results/{}_{}_{}_disp_ts_corr_inc.jpg'.format(insar_frame, gnss_station, ml_clip), format='jpg', dpi=400, bbox_inches='tight')
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

dates_dict = {}
for i, name in enumerate(imdates_dict):
    imdates = imdates_dict[name]
    dates_temp = []
    for date in imdates:
        date_str = str(date) 
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        dates_temp.append(date_obj)
    dates_dict[name] = dates_temp

slope_dict = {}
slope_plot_dict = {}
std_err_dict = {}
std_err_plot_dict = {}
regression_line_dict = {}
intercept_dict = {}

for i, name in enumerate(cum_subset_ts_dict):
    cum_use = cum_subset_ts_dict[name]
    dates_dec_use = dates_dec_subset_dict[name]
    dates_subset = dates_subset_dict[name]
    valid_indices = ~np.isnan(cum_use)
    dates_dec_subset = (np.array(dates_dec_use))[valid_indices]
    cum_subset = (np.array(cum_use))[valid_indices]
    dates_subset = (np.array(dates_subset))[valid_indices]
    slope_i, intercept_i, r_value_i, p_value_i, std_err_i = linregress(dates_dec_subset, cum_subset)
    slope_dict[name] = slope_i
    std_err_dict[name] = std_err_i
    intercept_dict[name] = intercept_i
    slope_plot_dict[name] = format(slope_i, '.2f')
    std_err_plot_dict[name] = format(std_err_i, '.1f')
    regression_line_dict[name] = slope_dict[name] * np.array(dates_dec_subset) + intercept_dict[name]

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
plt.plot(dates_subset_dict['Goldstein + Cascade'], cum_subset_ts_dict['Goldstein + Cascade'], color='blue', marker="o", label='N', linestyle='None', markersize=6, linewidth=0.5)
plt.plot(dates_subset_dict['Goldstein + Cascade'], regression_line_dict['Goldstein + Cascade'], 'g')
#plt.text(dfGPS.Dates.iloc[150], -60, 'Linear fit: ' + slope_plot_i + ' +/- ' + std_err_plot_i + ' mm/yr', fontsize=16)
ax3.set_title("Goldstein + Cascade {} InSAR LoS Displacement".format(insar_frame), fontsize=16)

ax4 = plt.subplot(4,1,4, sharex=ax1)
trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
ax4.text(0.0, 0.1, "d)", transform=ax4.transAxes + trans,fontsize='large', verticalalignment='top',     bbox=dict(facecolor='white', edgecolor='none', pad=3.0))
plt.plot(dates_subset_dict['No reunwrapping'], cum_subset_ts_dict['No reunwrapping'], color='blue', marker="o", label='N', linestyle='None', markersize=6, linewidth=0.5)
plt.plot(dates_subset_dict['No reunwrapping'], regression_line_dict['No reunwrapping'], 'g')
##plt.text(dfGPS.Dates.iloc[150], -60, 'Linear fit: ' + slope_plot_d + ' +/- ' + std_err_plot_d + ' mm/yr', fontsize=16)
ax4.set_title("No reunwrapping {} InSAR LoS Displacement".format(insar_frame), fontsize=16)

plt.xlabel('Dates', fontsize=18,fontweight='bold') 
plt.ylabel('mm', fontsize=18, x= -5)   
fig.suptitle("Displacement time-series at point {}, {}".format(gnss_lon,gnss_lat), fontweight='bold', fontsize=18, y=0.98)

plt.tight_layout()
#plt.savefig('/nfs/a285/homes/eejap/reunwrap_tests/078A_07049_131313/{}_{}_{}_disp_ts_lin_fit_corr_inc.jpg'.format(insar_frame, gnss_station, ml_clip), dpi=400, bbox_inches='tight')
plt.show()

#%% Start all time-series at 0 to compare
GPS_dLOS_first = GPS_dLOS.iloc[0]
GPS_dLOS_zero = GPS_dLOS - GPS_dLOS_first

dfGPS_dU_first = gnss_v_subset.iloc[0]
dfGPS_dU_zero = gnss_v_subset- dfGPS_dU_first

cum_zero_dict={}
for i, name in enumerate(cum_subset_ts_dict):
    cum_use = cum_subset_ts_dict[name]
    cum_first = cum_use[0]
    cum_zero_dict[name] = cum_use - cum_first

#%% Calculate new linear fit for zeroed vels
## Zero linear fits
me_line_zero = me_line - me_line.iloc[0]
regression_line_zero = regression_line - regression_line.iloc[0]

regression_line_zero_dict={}
for i, name in enumerate(regression_line_dict):
    line_use = regression_line_dict[name]
    regression_line_zero_dict[name] = line_use - line_use[0]

#%%
# try to plot midas line
gnss_dec_dates_subset_normalised = gnss_dec_dates_subset - gnss_dec_dates_subset.iloc[0]
midas_ydisplacement = np.median(MIDAS_los)*gnss_dec_dates_subset_normalised

#%% plot 'normalised' time-series
plt.figure(figsize=(12,8))
#plt.plot(gnss_dates_subset, midas_ydisplacement, label="{} LoS GNSS({} +/- {} mm/yr)".format(gnss_station, ), color = 'red')
MIDAS_los_plot = round(np.median(MIDAS_los), 2)
MIDAS_los_err_plot = round(MIDAS_los_err, 1)
plt.plot(gnss_dates_subset, midas_ydisplacement, label="GNSS ({} +/- {} mm/yr)".format(MIDAS_los_plot, MIDAS_los_err_plot), color='red', linewidth=3)

# write GNSS station name in top-right
#plt.text(gnss_dates_subset.iloc[1068], -1, gnss_station, weight='bold', fontsize=24) # ICMX
#plt.text(gnss_dates_subset.iloc[1041], -3, gnss_station, weight='bold', fontsize=24) # UTUL
#plt.text(gnss_dates_subset.iloc[2600], -40, gnss_station, weight='bold', fontsize=24) # MMX1

# define line colour dictionary
colours = ['mediumblue', 'teal', 'cornflowerblue', 'aquamarine']
colour_dict = dict(zip(titles_vels, colours))

# plot InSAR time-series
for i, name in enumerate(regression_line_zero_dict):
    plt.plot(dates_subset_dict[name], regression_line_zero_dict[name], label="{} ({} +/- {} mm/yr)".format(name, slope_plot_dict[name], std_err_plot_dict[name]), color = colour_dict[name], linewidth=3)

# place text below plot
#plt.figtext(0.5, 0.025, 'MIDAS LoS GNSS fit: ' + f"{round(np.median(MIDAS_los), 2)} +/- {round(MIDAS_los_err, 3)} mm/yr", fontsize=12, ha='center')
#for i, name in enumerate(regression_line_zero_dict):
    #plt.figtext(0.5, 0-0.025*i, 'LoS InSAR linear velocity with {}: '.format(name) + slope_plot_dict[name] + ' +/- ' + std_err_plot_dict[name] + ' mm/yr', fontsize=12, ha='center')

# Set y-axis properties to plot on the right side
ax = plt.gca()
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()

#plt.xlabel('Dates', fontsize=20) 
plt.ylabel('mm',fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=20)
#plt.title("{} LoS displacement time-series at {} ({}, {})".format(insar_frame, gnss_station, gnss_lon,gnss_lat), fontweight='bold')

plt.legend(loc='lower left', fontsize=16)

plt.savefig('/nfs/see-fs-02_users/eejap/public_html/sarwatch_paper/test_gnss_figs/{}_{}_{}_disp_ts_lin_fit_zeroed_corr_inc.jpg'.format(insar_frame, gnss_station, ml_clip), dpi=400, bbox_inches='tight')

#%% Plot Lat v los velocity

# create vel dict
vel_dict = dict(zip(titles_vels, vel_list))

# Define start and end points of profile (lon, lat)
start_point = (-99.18, 19.836)
end_point = (-98.98, 19.127)
#start_point = (-99.063, 19.809)
#end_point = (-98.87, 19.070)

# Create an array of indices for the file
plt.figure(figsize=(6,10))

# Loop through arrays in the dictionary and plot a profile for each
lons_dict={}
lats_dict={}

# find line for lat profile
m = (dfgnss_info.at['MMX1', 'lat']-dfgnss_info.at['UTUL', 'lat'])/(dfgnss_info.at['MMX1', 'lon']-dfgnss_info.at['UTUL', 'lon'])
c = dfgnss_info.at['MMX1', 'lat']-m*dfgnss_info.at['MMX1', 'lon']
#m = (dfgnss_info.at['MXTM', 'lat']-dfgnss_info.at['UFXN', 'lat'])/(dfgnss_info.at['MXTM', 'lon']-dfgnss_info.at['UFXN', 'lon'])
#c = dfgnss_info.at['MXTM', 'lat']-m*dfgnss_info.at['MXTM', 'lon']


for i, name in enumerate(vel_dict):
    vel_use = vel_dict[name]
    lats = m*np.linspace(start_point[0], end_point[0], np.min(vel_dict[name].shape))+c
    lons = (lats-c)/m
    lons_dict[name] = lons
    lats_dict[name] = lats

lats_indx_dict = {}
for i, name in enumerate(lats_dict):
    lats = lats_dict[name]
    lats_temp = []
    for lat in lats:
        lats_temp.append(np.abs(lat_rw - lat).argmin())
    lats_indx_dict[name] = lats_temp
      
lons_indx_dict = {}
for i, name in enumerate(lons_dict):
    lons = lons_dict[name]
    lons_temp = []
    for lon in lons:
        lons_temp.append(np.abs(lon_rw - lon).argmin())
    lons_indx_dict[name] = lons_temp
    
values_dict = {}
for i, name in enumerate(lats_indx_dict):
    lats_indx = lats_indx_dict[name]
    lons_indx = lons_indx_dict[name]
    values_temp = []
    values_temp.append(vel_dict[name][lats_indx_dict[name], lons_indx_dict[name]])
    values_dict[name] = np.array(values_temp, dtype=np.float64)

# Plot the profile
for i, name in enumerate(values_dict):
    plt.plot(np.squeeze(values_dict[name]), lats_dict[name], label="{}".format(name), color = colour_dict[name])
    
# plot gnss along profile
plt.plot(-190.84, dfgnss_info.at['MMX1', 'lat'], 'x', markersize=12, color='red')
plt.plot( -24.03, dfgnss_info.at['UTUL', 'lat'],'x', markersize=12, color='red')
#plt.plot(-186.95, dfgnss_info.at['MXTM', 'lat'], 'x', markersize=12, color='red')
#plt.plot( -88.86, dfgnss_info.at['UFXN', 'lat'],'x', markersize=12, color='red')

font = {"size": 14}

plt.text(-190.84-20, dfgnss_info.at['MMX1', 'lat']-0.028, 'MMX1', fontdict=font)
plt.text( -24.03-15, dfgnss_info.at['UTUL', 'lat']-0.028,'UTUL', fontdict=font)
#plt.text(-186.95-20, dfgnss_info.at['MXTM', 'lat']-0.028, 'MXTM', fontdict=font)
#plt.text( -88.86-15, dfgnss_info.at['UFXN', 'lat']-0.028,'UFXN', fontdict=font)
  
# Add legend and axis labels
plt.legend(loc='upper left', fontsize=12)
plt.ylabel('Latitude (deg)', fontsize=13)
plt.xlabel('LoS Velocity (mm/yr)', fontsize=13)
plt.tick_params(axis='both', which='major', labelsize=14)
#plt.title('{} LoS velocity profiles'.format(insar_frame))

# Set y-axis properties to plot on the right side
ax = plt.gca()
ax.yaxis.set_label_position("left")
ax.yaxis.tick_left()

plt.savefig('/nfs/see-fs-02_users/eejap/public_html/sarwatch_paper/{}_{}_rw_lat_profile_MMX1_ICMX.jpg'.format(insar_frame, ml_clip), dpi=400, bbox_inches='tight')

#%% Plot Lon v los velocity

start_point_horiz = (-99.2, 19.4)
end_point_horiz = (-98.87, 19.47)

# Create an array of indices for the file
plt.figure(figsize=(10,6))

# Loop through arrays in the dictionary and plot a profile for each
lons_dict={}
lats_dict={}

for i, name in enumerate(vel_dict):
    vel_use = vel_dict[name]
    lons, lats = np.linspace(start_point_horiz[0], end_point_horiz[0], np.min(vel_use.shape)), np.linspace(start_point_horiz[1], end_point_horiz[1], np.min(vel_use.shape))
    lons_dict[name] = lons
    lats_dict[name] = lats

lats_indx_dict = {}
for i, name in enumerate(lats_dict):
    lats = lats_dict[name]
    lats_temp = []
    for lat in lats:
        lats_temp.append(np.abs(lat_rw - lat).argmin())
    lats_indx_dict[name] = lats_temp
      
lons_indx_dict = {}
for i, name in enumerate(lons_dict):
    lons = lons_dict[name]
    lons_temp = []
    for lon in lons:
        lons_temp.append(np.abs(lon_rw - lon).argmin())
    lons_indx_dict[name] = lons_temp
    
values_dict = {}
for i, name in enumerate(lats_indx_dict):
    lats_indx = lats_indx_dict[name]
    lons_indx = lons_indx_dict[name]
    values_temp = []
    values_temp.append(vel_dict[name][lats_indx_dict[name], lons_indx_dict[name]])
    values_dict[name] = np.array(values_temp, dtype=np.float64)


# Plot the profile
for i, name in enumerate(values_dict):
    plt.plot(lons_dict[name], np.squeeze(values_dict[name]), label="{}".format(name), color = colour_dict[name])
    
# plot gnss along profile
plt.plot(dfgnss_info.at['MMX1', 'lon'], -190.84,'x', markersize=12, color='red')
plt.plot(dfgnss_info.at['ICMX', 'lon'], -13.39, 'x', markersize=12, color='red')

plt.text(dfgnss_info.at['MMX1', 'lon']-0.034, -190.84-3.5, 'MMX1', fontdict=font)
plt.text(dfgnss_info.at['ICMX', 'lon']-0.03, -13.39-3.5, 'ICMX', fontdict=font)
    
# Add legend and axis labels
plt.legend(loc='lower left', fontsize=10)
plt.xlabel('Longitude (deg)', fontsize=13)
plt.ylabel('LoS Velocity (mm/yr)', fontsize=13)
plt.tick_params(axis='both', which='major', labelsize=14)
#plt.title('{} LoS velocity profiles'.format(insar_frame))

# Set y-axis properties to plot on the right side
ax = plt.gca()
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()

plt.savefig('/nfs/a285/homes/eejap/reunwrap_tests/078A_07049_131313/results/{}_{}_rw_lat_profile.jpg'.format(insar_frame, ml_clip), dpi=400, bbox_inches='tight')

#%%
# Plot the 'vel' variable with latitude and longitude on the axes
plt.imshow(vel_dict['No reunwrapping'], extent=[lon_min_rw, lon_max_rw, lat_min_rw, lat_max_rw], cmap = cm.vik, vmin=-200, vmax=200)

#make dict for GNSS Sites
gnss_stations = ['MXTM', 'MMX1', 'MXMX', 'UFXN', 'UNVA', 'UTUL', 'ICMX']

# Add profile line
plt.plot([end_point[0], start_point[0]], [end_point[1], start_point[1]], color='red')
plt.plot([end_point_horiz[0], start_point_horiz[0]], [end_point_horiz[1], start_point_horiz[1]], color='red')

# Add a marker for the specific point (gnss_lat, gnss_lon)
for x in gnss_stations:
    plt.plot(dfgnss_info.at[x, 'lon'], dfgnss_info.at[x, 'lat'], 'rx', markersize=8)

# Add labels and title to the plot
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('{} Gaussian {} + {}'.format(insar_frame, ml_clip, gnss_station))
colorbar = plt.colorbar()
colorbar.set_label('LoS Velocity (mm/yr)')
plt.savefig('/nfs/a285/homes/eejap/reunwrap_tests/078A_07049_131313/results/{}_{}_{}.reunw_insar_los_map_MMX1_ICMX.jpg'.format(insar_frame, gnss_station, ml_clip), dpi=400, bbox_inches='tight')
plt.show()


#%%
# # Add a marker for the specific point (gnss_lat, gnss_lon)
# plt.imshow(vel_def, extent=[lon_min_def, lon_max_def, lat_min_def, lat_max_def], cmap = cm.vik, vmin=-200, vmax=200)
# plt.plot(gnss_lon, gnss_lat, 'rx', markersize=8)

# plt.plot([end_point[0], start_point[0]], [end_point[1], start_point[1]], color='red')
# plt.plot([end_point_horiz[0], start_point_horiz[0]], [end_point_horiz[1], start_point_horiz[1]], color='red')

# # Add labels and title to the plot
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.title('{} {} + {}'.format(insar_frame, ml_clip, gnss_station))
# colorbar = plt.colorbar()
# colorbar.set_label('LoS Velocity (mm/yr)')
# #plt.savefig('./outputs/{}/maps/{}_{}_{}.insar_los_map.jpg'.format(insar_frame, insar_frame, gnss_station, ml_clip), dpi=400, bbox_inches='tight')
# plt.show()