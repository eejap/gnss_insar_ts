#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 16:12:04 2023

@author: eejap
"""

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

# this is Andrew Watson's library of functions, see https://github.com/Active-Tectonics-Leeds/interseismic_practical
import sys
sys.path.append('/nfs/a285/homes/eejap/plots/gnss_insar_ts')
import interseis_lib as lib

#%%
directorio= '/nfs/a285/homes/eejap/plots/gnss_insar_ts'
gnss_station = "MXTM"
filename = directorio + '/data/gnss/' + gnss_station + '.NA.tenv3'
gnss_station_info = directorio + '/data/gnss/gnss_info.txt'
formatSpec = '%4C%8s%10f%6f%5f%2f%7f%7f%10f%10f%10f%6f%10f%8f%9f%9f%9f%10f%10f%f%[^\n\r]'

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

#%%
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

#%%
# Extract the valid values for each component
E1yr = pairs[np.isfinite(pairs[:, 0]), 0]
N1yr = pairs[np.isfinite(pairs[:, 1]), 1]
U1yr = pairs[np.isfinite(pairs[:, 2]), 2]

# Plot the histograms
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

axs[0].hist(E1yr)
axs[0].set_title('Median vel: ' + f'{round(np.median(E1yr),3)} mm/yr')
axs[0].set_xlabel('E velocity (mm/yr)')
axs[0].set_ylabel('Binned freq')

axs[1].hist(N1yr)
axs[1].set_title('Median vel: ' + f'{round(np.median(N1yr),3)} mm/yr')
axs[1].set_xlabel('N velocity (mm/yr)')

axs[2].hist(U1yr)
axs[2].set_title('Median vel: ' + f'{round(np.median(U1yr),3)} mm/yr')
axs[2].set_xlabel('U velocity (mm/yr)')

plt.show()

# %% MIDAS Approach - Estimate dispersion and Trim Tails to re-estimate
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
#%%
# Plot the histograms
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

axs[0].hist(E1yrtrim)
axs[0].set_title('Median Error: ' + f'{round(np.median(E1yrtrimerr),3)} mm/yr')
axs[0].set_xlabel('E error (mm/yr)')
axs[0].set_ylabel('Binned freq')

axs[1].hist(N1yrtrim)
axs[1].set_title('Median Error: ' + f'{round(np.median(N1yrtrimerr),3)} mm/yr')
axs[1].set_xlabel('N error (mm/yr)')

axs[2].hist(U1yrtrim)
axs[2].set_title('Median Error: ' + f'{round(np.median(U1yrtrimerr),3)} mm/yr')
axs[2].set_xlabel('U error (mm/yr)')

plt.show()

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

# %% Plot Velcoity and Uncertianty

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].plot(date, E, 'x')
axs[0].set_title('MIDAS fit + error: ' + f"{round(np.median(E1yr),3)} +/- {round(E1yrtrimstderr_midas,3)} mm/yr")
axs[0].set_xlabel('Year')
axs[0].set_ylabel('Displacement (mm)')

axs[1].plot(date, N, 'x')
axs[1].set_title('MIDAS fit + error: ' + f"{round(np.median(N1yr),3)} +/- {round(N1yrtrimstderr_midas,3)} mm/yr")
axs[1].set_xlabel('Year')

axs[2].plot(date, U, 'x')
axs[2].set_title('MIDAS fit + error: ' + f"{round(np.median(U1yr),3)} +/- {round(U1yrtrimstderr_midas,3)} mm/yr")
axs[2].set_xlabel('Year')

# Adjust the spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()