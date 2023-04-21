# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 22:43:50 2023

@author: jakep
"""

import numpy as np
from scipy.optimize import minimize

# Define the positions of the three recording devices
device1 = np.array([15, 0])
device2 = np.array([-15, 0])
device3 = np.array([15, 11.5])

# Record the times at which the event occurs on each device
ref = 41.17013605
time1 = 41.26956916 - ref # time in seconds
time2 = 41.17013605 - ref
time3 = 41.97247166 - ref
print(time1, time2, time3)
# Calculate the distances from the event to each device
speed_of_sound = 1125.33 # ft/s 343 # meters per second
distance1 = speed_of_sound * time1
distance2 = speed_of_sound * time2
distance3 = speed_of_sound * time3

# Define a function that calculates the error between the predicted and actual distances
def error(point):
    d1 = np.linalg.norm(point - device1)
    d2 = np.linalg.norm(point - device2)
    d3 = np.linalg.norm(point - device3)
    return abs(d1 - distance1) + abs(d2 - distance2) + abs(d3 - distance3)

# Use the minimize function to find the point that minimizes the error
initial_guess = np.array([1, 1])  # starting point for optimization
result = minimize(error, initial_guess)

# Print the estimated position of the event
print("Estimated position:", result.x)