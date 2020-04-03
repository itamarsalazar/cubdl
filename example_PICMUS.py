# File:       example_PICMUS.py
# Author:     Dongwoon Hyun (dongwoon.hyun@stanford.edu)
# Created on: 2020-03-12
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from das_torch import DAS_torch
from PlaneWaveData import PICMUSData
from PixelGrid import make_pixel_grid

# Load PICMUS dataset
database_path = "../datasets/picmus"
acq = "simulation"
target = "contrast_speckle"
dtype = "iq"
P = PICMUSData(database_path, acq, target, dtype)

# Define pixel grid limits (assume y == 0)
xlims = [P.ele_pos[0, 0], P.ele_pos[-1, 0]]
zlims = [5e-3, 55e-3]
wvln = P.c / P.fc
dx = wvln / 4
dz = dx  # Use square pixels
grid = make_pixel_grid(xlims, zlims, dx, dz)
fnum = 1

# Make 75-angle image
IN, QN = DAS_torch(P, grid)
IQN = (IN + 1j * QN).T  # Tranpose for display purposes
bimgN = 20 * np.log10(np.abs(IQN))  # Log-compress
bimgN -= np.amax(bimgN)  # Normalize by max value

# Make 1-angle image
idx = len(P.angles) // 2  # Choose center angle
I1, Q1 = DAS_torch(P, grid, idx)
IQ1 = (I1 + 1j * Q1).T  # Transpose for display purposes
bimg1 = 20 * np.log10(np.abs(IQ1))  # Log-compress
bimg1 -= np.amax(bimg1)  # Normalize by max value

# Display images via matplotlib
extent = [xlims[0] * 1e3, xlims[1] * 1e3, zlims[1] * 1e3, zlims[0] * 1e3]
plt.subplot(121)
plt.imshow(bimgN, vmin=-60, cmap="gray", extent=extent, origin="upper")
plt.title("%d angles" % len(P.angles))
plt.subplot(122)
plt.imshow(bimg1, vmin=-60, cmap="gray", extent=extent, origin="upper")
plt.title("Angle %d: %ddeg" % (idx, P.angles[idx] * 180 / np.pi))
plt.show()
