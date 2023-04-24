# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 23:05:09 2023
Sound Location Project

@author: jakep
"""
# %% Imports
import numpy as np
import matplotlib.pyplot as plt
# import sympy as sym
# from sympy import plot_implicit, Eq
from scipy.io import wavfile
from scipy.signal import find_peaks
# from scipy.spatial import distance
# from scipy.optimize import curve_fit


#%% Load Data
# Load wav files into np arrays
samplerate, kevin_phone = wavfile.read('kevin_phone.wav')
_, jake_phone = wavfile.read('jake_phone.wav')
_, curtis_phone = wavfile.read('curtis_phone.wav')

wavs = [kevin_phone, jake_phone, curtis_phone]
names = ['k_phone', 'j_phone', 'c_phone']

#%% Constants/Helpers
# Constants
# speed_sound = 343 # m/s
speed_sound = 1125.33 # ft/s
sync_distanceKJ = 4.572 # m (15 ft)
sync_distanceKC = 3.505 # m (11.5 ft)

# Helpers
def filter_peaks(peaks, sample_sep):
    peaks_filtered = []
    prev_peak = peaks[0]
    for peak in peaks[1:]:
        if peak - prev_peak > sample_sep:
            peaks_filtered.append(prev_peak)
            prev_peak = peak
    peaks_filtered.append(prev_peak)
    return np.array(peaks_filtered)

def make_hyp(a, b):  
    return np.array(np.sqrt((x**2/a**2 - 1)*b**2))

def plot_hyp(ax, hyp, label):
    ax.plot(x, hyp, label=label)
    # ax.plot(x, -hyp, label=f'-{label}')
    ax.grid()
    ax.legend()

def rot_hyp(hyp_pts, theta):
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    data = rot_matrix.dot(np.column_stack((x, hyp_pts)).T)
    return data

def distance_btw(coord1, coord2=(0,0)):
    return np.sqrt((coord1[0]-coord2[0])**2 + (coord1[1]-coord2[1])**2)
#%% Find Peaks Routine
# Use scipy to find peaks in each signal above selected threshold
threshold = 5000
k_peaks, _ = find_peaks(kevin_phone, height=threshold)
j_peaks, _ = find_peaks(jake_phone, height=threshold)
c_peaks, _ = find_peaks(curtis_phone, height=threshold)

# Filter Peaks by selected sample separation
sample_sep = 0.15e2
kpeaks_filtered = filter_peaks(k_peaks, sample_sep)
jpeaks_filtered = filter_peaks(j_peaks, sample_sep)
cpeaks_filtered = filter_peaks(c_peaks, sample_sep)

# Find sync sample diff
s_syncKJ = kpeaks_filtered[0] - jpeaks_filtered[0]
s_syncKC = abs(kpeaks_filtered[2] - cpeaks_filtered[2])
print(s_syncKC, s_syncKJ)

ksync = kevin_phone[s_syncKJ:]
csync = curtis_phone[s_syncKC:]

sync_wavs = [ksync, jake_phone, csync]
#%% Selecting Claps from filtered signal peaks
clap_timestamps = np.array([7, 17.5, 40.5, 81, 131]) # approx times, in seconds, around peaks of interest
claps_labels = ['sync1', 'sync2', 'clap1','clap2','clap3']
claps = {}
# This is a bit ugly, but tuned to get the data we want right now...
# k [0]. j[1], c[2]
for i, ts in enumerate(clap_timestamps):
    tmp_k = np.argmax((kpeaks_filtered >= ts*samplerate) & (kpeaks_filtered < ts*samplerate+0.5e6))
    tmp_j = np.argmax((jpeaks_filtered >= ts*samplerate) & (jpeaks_filtered < ts*samplerate+0.5e6))
    tmp_c = np.argmax((cpeaks_filtered >= ts*samplerate) & (cpeaks_filtered < ts*samplerate+0.5e6))
    claps[claps_labels[i]] = [kpeaks_filtered[tmp_k], jpeaks_filtered[tmp_j], cpeaks_filtered[tmp_c]]

# confirm syncs are the same
# print(s_syncKJ == claps['sync1'][0] - claps['sync1'][1])
# print(s_syncKC == abs(claps['sync2'][0] - claps['sync2'][2]))

s_syncKJ = claps['sync1'][0] - claps['sync1'][1]
s_syncKC = abs(claps['sync2'][0] - claps['sync2'][2])
#%% Find parameter 'a' from synced signals
# Find the diff (num samples) between synced signals
diffs_KJ = []
diffs_KC = []

for label, value in claps.items():
    diffs_KJ.append(abs(value[0] - value[1] - s_syncKJ))
    diffs_KC.append(abs(value[2] - s_syncKC -value[0]))
    
diffsKJ_s = np.array(diffs_KJ[2:]) / samplerate
diffsKC_s = np.array(diffs_KC[2:]) / samplerate

a_KJ = 0.5 * diffsKJ_s * speed_sound
a_KC = 0.5 * diffsKC_s * speed_sound

#%% Hyperbola Parameters
# eq: x^2/a^2 - y^2/b^2 = 1
# b^2 = c^2 - a^2, where c is distance from phone to origin, 15 ft
c_KJ = 15 # ft
c_KC = 11.5 # ft
b_KJ = np.sqrt(abs(c_KJ**2 - a_KJ**2))
b_KC = np.sqrt(abs(c_KC**2 - a_KC**2)) 

#%% Generate points
theta = np.radians(92.9)
lim = 35
N = 1000
x = np.linspace(-lim, lim, N)

# Clap 1
hyp1_KJ = make_hyp(a_KJ[0],b_KJ[0])
hyp1_KC = make_hyp(a_KC[0], b_KC[0])
rot_KC = rot_hyp(hyp1_KC, theta)

# Clap 2
hyp2_KJ = make_hyp(a_KJ[1],b_KJ[1])
hyp2_KC = make_hyp(a_KC[1], b_KC[1])
rot_KC2 = rot_hyp(hyp2_KC, theta)

#%% Hyperbolas Plot
figh, (h1, h2, h3) = plt.subplots(1, 3, figsize=(14,6))

h1.plot(rot_KC[0]+c_KJ,x+c_KC, label='KC rot')
# h1.plot(-data[0]+15, x+11.5)
h1.set_xlim(-lim, lim)
h1.set_ylim(-30, 30)
plot_hyp(h1, hyp1_KJ, 'KJ')
h1.set_title('clap 1')

h2.plot(rot_KC2[0]+c_KJ,x+c_KC, label='KC rot')
h2.set_xlim(-lim, lim)
h2.set_ylim(-30, 30)
plot_hyp(h2, hyp2_KJ, 'KJ')
h2.set_title('clap 2')

# h3.plot(x, y(a_KJ[2],b_KJ[2]), label='KJ')
# h3.plot(x, -(y(a_KJ[2],b_KJ[2])),label='KJ')
# h3.plot(x, y(a_KC[2], b_KC[2]), label='KC')
# h3.plot(x, -y(a_KC[2], b_KC[2]), label='KC')
# h3.set_title('clap 3')
# h3.grid()
# h3.legend()

figh.tight_layout()

#%% Scratch
# attempt at rotate plot
# define the hyperbolas
X = x
Y1 = make_hyp(a_KJ[2], b_KJ[2])
Y2 = make_hyp(a_KC[2], b_KC[2])

# rotate the second hyperbola
theta = np.radians(92.91)
rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
coords = np.vstack((X, Y2)).T
coords_rotated = np.dot(coords, rot_matrix)
X2 = coords_rotated[:, 0] + 15
Y2 = coords_rotated[:, 1] + 11.5

# plot the hyperbolas
plt.plot(X, Y1, color='red', label='KJ')
plt.plot(X2, Y2, color='blue', label='KC')
plt.scatter([-15, 15, 15], [0, 0, 23], color='black')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

#%% Plots
# Quick Plot
qfig, ax = plt.subplots(1, figsize=(11, 7))

# ax.set_xlim((1.81e6, 1.89e6))
ax.plot(ksync, label='k_phone', alpha = 0.5)
ax.plot(csync, label='c_phone', alpha=0.7, ls=':')
ax.plot(jake_phone, label='j_phone', ls='--', alpha=0.7)
for label, clap in claps.items():
    # tmp_str = f'{label}'
    for i, c in enumerate(clap):
        ax.scatter(c, sync_wavs[i][c])

ax.legend()

#%%# Plot all 3 signals together
fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(11,9))
fig.suptitle('All 3 Signals')
for n, wav in enumerate(wavs):
    ax1.plot(wav, label=names[n])
ax1.legend()
# Plot synced signals separately below
ax2.plot(kevin_phone[s_syncKJ:], label='k_phone')
ax2.plot(jake_phone, label='j_phone')
ax2.legend()


ax3.plot(kevin_phone, label='k_phone')
ax3.plot(curtis_phone[s_syncKC:], label='c_phone')
ax3.plot(jake_phone[s_syncKJ:], label='j_phone', ls='--', alpha=0.7)
ax3.legend()
