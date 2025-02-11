#!/usr/bin/env python

from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import numpy as np

#	1st (preferred) approach to center and scale data.
X = np.array([[3, 0, 400], [2, 1, 200], [2, 2, 500], [1, 1, 100]])

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler to the data and transform the data
scaled_data = scaler.fit_transform(X)

print("Original Data:\n", X)
print("Centered and Scaled Data:\n", scaled_data)



#	2nd approach to center and scale data.
#	----------------------------
# Original data
X = np.array([[3, 0, 400], [2, 1, 200], [2, 2, 500], [1, 1, 100]])
print("Original Data:\n", X)

# Scale the data
X_scaled = preprocessing.scale(X)
print("\nScaled Data:\n", X_scaled)

# Verify the mean and standard deviation of the scaled data
print("\nMean of Scaled Data (should be close to 0):\n", np.mean(X_scaled, axis=0))
print("\nStandard Deviation of Scaled Data (should be close to 1):\n", np.std(X_scaled, axis=0))
