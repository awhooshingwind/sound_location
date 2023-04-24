# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 00:16:55 2023

process_signal
a routine for processing wav recordings for
phoRan-C sound location

@author: jakep
"""
#%% Imports
import os
import tkinter as tk
from tkinter import filedialog

import numpy as np
import matplotlib.pyplot as plt

from scipy.io import wavfile
from scipy.signal import find_peaks, resample_poly

#%% Setup
def load_multiple_files_dialog():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_paths = filedialog.askopenfilenames(title="Select files to load")
    return file_paths

def extract_file_name(file_path):
    file_name = os.path.basename(file_path)
    file_name_without_extension, _ = os.path.splitext(file_name)
    return file_name_without_extension

#%% Load Data
files = load_multiple_files_dialog()

data = {}
names = []
for i, file in enumerate(files):
    names.append(extract_file_name(file))
    samplerate, data[f'{names[i]}'] = wavfile.read(file)
    
#%% Process
peaks = {}

for key in data.keys():
    peaks[f'{key}1'],_ = find_peaks(data[key][:,0], height=3000)
    peaks[f'{key}2'],_ = find_peaks(data[key][:,1], height=3000)

# plt.scatter(peaks['comp1'], data['comp'][peaks['comp1']][:,0])

