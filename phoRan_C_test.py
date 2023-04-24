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
from scipy.signal import find_peaks, resample_poly

#%% Load Recordings
samplerate, phone = wavfile.read('phone.wav')
_, comp = wavfile.read('comp.wav')

#%% Peaks
peaks = {}
peaks['phone'],_ = find_peaks(phone[:,0], height=5000)
peaks['comp'],_ = find_peaks(comp[:,0], height=5000)
# #%% Sync
sync = abs(peaks['comp'][0] - peaks['phone'][0])
# comp = comp[0:][sync:]
# %% Trim
max_len = len(phone)
comp = comp[0:][0:max_len+sync]

# left_phone = phone[:,0]
# right_phone = phone[:,1]
# phones = [left_phone, right_phone]

# left_comp = comp[:,0]
# right_comp = comp[:,1]
# comps = [left_comp, right_comp]
#%% 
data = comp[0:][sync:]

length = data.shape[0] # / samplerate # for time
time = np.linspace(0., length, data.shape[0])
plt.figure(figsize=(13,9))

# plt.xlim(10, 23)
# plt.xlabel("Time [s]")
# plt.ylabel("Amplitude")
plt.plot(time, data[:,0], label="comp", ls='--',alpha=0.7)
plt.plot(time, phone[:,0], label = 'phone', alpha=0.6)
plt.plot(time, data[:,1], label='comp_R', ls='--',alpha=0.7)
plt.plot(time, phone[:,1], label='phone_R', alpha=0.6)
plt.legend()