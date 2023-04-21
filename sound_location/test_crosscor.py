# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 00:29:25 2023

@author: jakep
"""

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
# import sympy as sym
# from sympy import plot_implicit, Eq
from scipy.io import wavfile
import scipy.signal as signal
from scipy.signal import find_peaks, correlate
from scipy.optimize import curve_fit


#%% Load Data
# Load wav files into np arrays
samplerate, kevin_phone = wavfile.read('kevin_phone.wav')
_, jake_phone = wavfile.read('jake_phone.wav')
_, curtis_phone = wavfile.read('curtis_phone.wav')

wavs = [kevin_phone, jake_phone, curtis_phone]
names = ['k_phone', 'j_phone', 'c_phone']

k_timestamps, j_timestamps, c_timestamps = [np.arange(len(wav)) / samplerate for wav in wavs]
#%% Constants/Helpers
# Constants
# speed_sound = 343 # m/s
speed_sound = 1125.33 # ft/s
sync_distanceKJ = 4.572 # m (15 ft)
sync_distanceKC = 3.505 # m (11.5 ft)

# Helpers
def find_tdoa(sig1, sig2, fs):
    # Compute the cross-correlation of the two signals
    corr = signal.correlate(sig1, sig2, mode='full')
    delay = np.argmax(corr) - (len(sig1) - 1)

    # Convert delay to time (seconds)
    tdoa = delay / fs

    return tdoa

#%% Slice then correlate approach
recordings = wavs
reference_idx = 0

# Find the time shift for each recording relative to the reference recording
time_shifts = []

for i, recording in enumerate(recordings):
    print(f'starting correlate loop, round {i}')
    if i == reference_idx:
        time_shifts.append(0)
    else:
        xcorr = correlate(recordings[reference_idx], recording)
        time_shift = np.argmax(xcorr) - len(recordings[reference_idx]) + 1
        time_shifts.append(time_shift)
print('correlated')
# Adjust the timestamps of the recordings
for i in range(len(recordings)):
    recordings[i].timestamps += time_shifts[i] / samplerate

# Assume we have three audio signals `sigA`, `sigB`, and `sigC`, and their respective
# sample rates `fsA`, `fsB`, and `fsC`.
sigK = kevin_phone
sigJ = jake_phone
sigC = curtis_phone
fsA = samplerate

# # Find the TDOA between A and B
# tdoa_KJ = find_tdoa(sigK, sigJ, fsA)

# # Find the TDOA between A and C
# tdoa_KC = find_tdoa(sigK, sigC, fsA)

# # Correct the timestamps of sigJ and sigC
# j_ts_corrected = j_timestamps - tdoa_KJ
# c_ts_corrected = c_timestamps - tdoa_KC