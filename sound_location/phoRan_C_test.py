# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 21:01:36 2023

PhoRan-C
Sound location project
@author: jakep
"""

#%% Imports
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import wavfile
from scipy.signal import find_peaks

#%% #Load recordings
samplerate, kevin_phone = wavfile.read('kevin_phone.wav')
_, jake_phone = wavfile.read('jake_phone.wav')
_, curtis_phone = wavfile.read('curtis_phone.wav')

wavs = [kevin_phone, jake_phone, curtis_phone]
names = ['k_phone', 'j_phone', 'c_phone']

#%% Quick plot
#%% Plot all 3
fig, ax = plt.subplots(1, figsize=(11,8))
left_lim = 5.92e6
right_lim = 6.1e6
ax.set_xlim(left_lim, right_lim)
for i, wav in enumerate(wavs):
    ax.plot(wav, label=f'{names[i]}')
    ax.legend()
    ax.grid()

#%% Setup
# Define Constants
speed_sound = 1125.33 # ft/s

# Helpers
def trim_signal(signal, min, max):
    return signal[int(min):int(max)]

def keep_peaks(signal_region, peak_height, width=1e4):
    tmp_k = {}
    for i, sig in enumerate(signal_region):
        ref_samp = 0
        keep = []
        tmp,_ = find_peaks(sig, height=peak_height)
        for t in tmp:
            if t >= ref_samp+width:
                ref_samp = t
                keep.append(t)
        tmp_k[f'{names[i]}'] = np.array(keep)
        
    return tmp_k

#%% Trim signals
sync_region = []
clap1 = []
clap2 = []
clap3 = []
clap4 = []
peak_h = 4000

for wav in wavs:
    sync_region.append(trim_signal(wav, 0, 0.93e6))
    clap1.append(trim_signal(wav, 1.79e6, 1.89e6))
    clap2.append(trim_signal(wav, 3.4e6, 3.65e6))
    clap3.append(trim_signal(wav, 5.3e6, 5.51e6))
    clap4.append(trim_signal(wav, 5.92e6, 6e6))

# sync_peaks = {}
# for i, sync in enumerate(sync_region):
#     ref_samp = 0
#     keep = []
#     tmp,_ = find_peaks(sync, height=peak_h)
#     for t in tmp:
#         if t >= ref_samp+1.3e4:
#             ref_samp = t
#             keep.append(t)
#     sync_peaks[f'{names[i]}'] = keep

sync_peaks = keep_peaks(sync_region, peak_h)
clap1_peaks = keep_peaks(clap1, peak_h)
clap2_peaks = keep_peaks(clap2, 1200)
clap3_peaks = keep_peaks(clap3, 1000, width = 6e4)
clap4_peaks = keep_peaks(clap4, peak_h)
#%% Plot all 3
fig, ax = plt.subplots(5,1, figsize=(14,9))
fig.tight_layout()
# left_lim = 0
# right_lim = 0.9e6
# ax.set_xlim(left_lim, right_lim)
def plot_region(ax, region, reg_peaks):
    for i, sig in enumerate(region):
        ax.plot(sig, label=f'{names[i]}')
        ax.legend()
        ax.grid()
    for i, rp in enumerate(reg_peaks.values()):
        ax.scatter(rp, wavs[i][rp], s=200)

plot_region(ax[0], sync_region, sync_peaks)
plot_region(ax[1], clap1, clap1_peaks)
plot_region(ax[2], clap2, clap2_peaks)
plot_region(ax[3], clap3, clap3_peaks)
plot_region(ax[4], clap4, clap4_peaks)