#!/usr/bin/env python

from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import numpy as np

#	1st (preferred) approach to center and scale data.
X = np.array([[3, 0, 400], [2, 1, 200], [2, 2, 500], [1, 1, 100]])
scaler = StandardScaler() # Initialize the StandardScaler
scaled_data = scaler.fit_transform(X)

print("Original Data:\n", X)
print("Centered and Scaled Data:\n", scaled_data)



#	2nd approach to center and scale data.
#	----------------------------
X_scaled = preprocessing.scale(X) # alternative to scale data
print("\nScaled Data:\n", X_scaled)

# Verify the mean and standard deviation of the scaled data
print("\nMean of Scaled Data (should be close to 0):\n", np.mean(X_scaled, axis=0))
print("\nStandard Deviation of Scaled Data (should be close to 1):\n", np.std(X_scaled, axis=0))
