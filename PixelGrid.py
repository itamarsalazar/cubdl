# File:       PixelGrid.py
# Author:     Dongwoon Hyun (dongwoon.hyun@stanford.edu)
# Created on: 2020-04-03
import numpy as np


def make_pixel_grid(xlims, zlims, dx, dz):
    """ Generate a pixel grid based on input parameters. """
    eps = 1e-10
    x = np.arange(xlims[0], xlims[1] + eps, dx)
    z = np.arange(zlims[0], zlims[1] + eps, dz)
    xx, zz = np.meshgrid(x, z, indexing="ij")
    yy = 0 * xx
    grid = np.stack((xx, yy, zz), axis=-1)
    return grid
