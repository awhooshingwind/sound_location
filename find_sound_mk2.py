# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 21:01:36 2023
Version 2 find_sound
Sound location project
@author: jakep
"""

#%% Imports
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import wavfile
from scipy.signal import find_peaks

#%% Load recordings
samplerate, kevin_phone = wavfile.read('kevin_phone.wav')
_, jake_phone = wavfile.read('jake_phone.wav')
_, curtis_phone = wavfile.read('curtis_phone.wav')

wavs = [kevin_phone, jake_phone, curtis_phone]
names = ['k_phone', 'j_phone', 'c_phone']

#%% Quick plot all 3
fig, ax = plt.subplots(1, figsize=(11,8))
left_lim = 1e6
right_lim = 2e6
ax.set_xlim(left_lim, right_lim)
for i, wav in enumerate(wavs):
    ax.plot(wav, label=f'{names[i]}')
    ax.legend()
    ax.grid()

#%% Setup
# Define Constants
speed_sound = 1125.33 # ft/s
cKJ = 15 # ft, distance from sync1 to either phone
cKC = 11.5 # ft, dist from sync2
cJC = 18.45 # ft, approx assuming right triangle between all phones

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

#%% Trim signals into event 'regions'
# Each region corresponds to one experimental config/sound event
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

sync_peaks = keep_peaks(sync_region, peak_h)
clap1_peaks = keep_peaks(clap1, peak_h)
clap2_peaks = keep_peaks(clap2, 1200)
clap3_peaks = keep_peaks(clap3, 1000, width = 6e4)
clap4_peaks = keep_peaks(clap4, peak_h)

regions = [
    sync_peaks,
    clap1_peaks,
    clap2_peaks,
    clap3_peaks,
    clap4_peaks,
    ]
#%% Plot all regions/peaks
fig, ax = plt.subplots(5,1, figsize=(14,9))
fig.tight_layout()

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

#%% Locate
sync1 = int((regions[0]['k_phone'] - regions[0]['j_phone'])[0])
sync2 = int(np.mean((regions[0]['c_phone'] - regions[0]['k_phone'])[1:]))
diffs = {}
str_label = 'clap'
for i,reg in enumerate(regions[1:]):
        tmp_KC = abs(reg['k_phone'] - reg['c_phone'] + sync2)
        tmp_KJ = abs(reg['k_phone'] - reg['j_phone'] - sync1)
        tmp_JC = abs(reg['j_phone'] - reg['c_phone'] + sync2 + sync1)
        tmp_tot = [tmp_KC, tmp_KJ, tmp_JC]
        diffs[f'{str_label}{i+1}'] = tmp_tot

#%% Quick plot all 3, sync
fig, ax = plt.subplots(1, figsize=(11,8))
left_lim = 1.8e6
right_lim = 1.9e6
ax.set_xlim(left_lim, right_lim)
ax.plot(kevin_phone[sync1:], label='ksync', alpha=0.6)
ax.plot(jake_phone, label='j_phone', ls=':', alpha=0.8)
ax.plot(curtis_phone[sync2+sync1:], label='csync', ls='--')
ax.grid()
ax.legend()

#%% Determine Hyperbola Parameters
labels = ['KC', 'KJ', 'JC']

c_vals = [cKC, cKJ, cJC]
a_vals = {}
b_vals = {}
for key in diffs.keys():
    for i in range(len(diffs[key])):
        tmp_a = (diffs[key][i]/samplerate) * 0.5 * speed_sound
        tmp_b = abs(c_vals[i]**2-tmp_a**2)
        a_vals[f'{key}_{labels[i]}'] = tmp_a
        b_vals[f'{key}_{labels[i]}'] = tmp_b

#%% Make Hyperbola(s)
def hyperbola(a, b):
    return np.sqrt((x**2/a**2 - 1)*b)

x = np.linspace(-50, 50, 1000)
hyp = {}

for key, value in a_vals.items():
    for i, _ in enumerate(value):
        hyp[key] = hyperbola(value[i], b_vals[key][i])

#%% Transforms
def rotation_matrix_2d(angle):
    """
    Create a 2D rotation matrix for a given angle.

    Args:
    angle (float): The angle of rotation in radians.

    Returns:
    numpy.ndarray: The 2x2 rotation matrix.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s],
                     [s, c]])

theta = np.radians(87.09)
thetaJC = np.radians(39.94)
rot_matrix_2d = rotation_matrix_2d(theta)
rot_matrixJC = rotation_matrix_2d(thetaJC)

clap1KC = np.array([x, hyp['clap1_KC']])
clap1JC = np.array([x, hyp['clap1_JC']])
clap2KC = np.array([x, hyp['clap2_KC']])
clap2JC = np.array([x, hyp['clap2_JC']])
clap3KC = np.array([x, hyp['clap3_KC']])
clap3JC = np.array([x, hyp['clap3_JC']])
clap4KC = np.array([x, hyp['clap4_KC']])
clap4JC = np.array([x, hyp['clap4_JC']])

rot_test = np.matmul(rot_matrix_2d, clap1KC)
rot3_test = np.matmul(rot_matrixJC, clap1JC)
rot_test2 = np.matmul(rot_matrix_2d, clap2KC)
rot4_test = np.matmul(rot_matrixJC, clap2JC)
rot_test3 = np.matmul(rot_matrix_2d, clap3KC)
rot5_test = np.matmul(rot_matrixJC, clap3JC)
rot_test4 = np.matmul(rot_matrix_2d, clap4KC)
rot6_test = np.matmul(rot_matrixJC, clap4JC)
# print(rot_test)
#%% Plotting
figh, ax = plt.subplots(4, 1, figsize=(13,9))
figh.tight_layout()
ax[0].plot(x, hyp['clap1_KJ'], label='KJ')
ax[0].plot(rot_test[0]+15, rot_test[1]+11.5, label='rot_KC')
ax[0].set_xlim(-30, 30)
ax[0].set_ylim(-10, 30)
ax[0].plot(rot3_test[0]+3.45, rot3_test[1]+11.5)
ax[0].scatter([-15, 15, 15],[0, 0, 23])
ax[0].grid()

ax[1].plot(x, hyp['clap2_KJ'], label='KJ')
ax[1].plot(rot_test2[0]+15, rot_test2[1]+11.5, label='rot_KC')
ax[1].set_xlim(-30, 30)
ax[1].set_ylim(-10, 30)
ax[1].plot(rot4_test[0]+3.45, rot4_test[1]+11.5)
ax[1].scatter([-15, 15, 15],[0, 0, 23])
ax[1].grid()

ax[2].plot(x, -hyp['clap3_KJ'], label='KJ')
ax[2].plot(rot_test3[0]+15, rot_test3[1]+11.5, label='rot_KC')
ax[2].set_xlim(-30, 30)
ax[2].set_ylim(-30, 30)
ax[2].plot(-(rot5_test[0]+3.45), -(rot5_test[1]+11.5))
ax[2].scatter([-15, 15, 15],[0, 0, 23])
ax[2].grid()

ax[3].plot(x, -hyp['clap4_KJ'], label='KJ')
ax[3].plot(rot_test4[0]+15, rot_test4[1]+11.5, label='rot_KC')
ax[3].set_xlim(-30, 30)
ax[3].set_ylim(-30, 30)
ax[3].plot(-(rot6_test[0]+3.45), -(rot6_test[1]+11.5))
ax[3].scatter([-15, 15, 15],[0, 0, 23])
ax[3].grid()

# for i, _ in enumerate(a_vals):
#     print(_, a_vals[_], b_vals[_])
#     ax.plot(x, (hyperbola(a_vals[_][0], b_vals[_][0])))
     
