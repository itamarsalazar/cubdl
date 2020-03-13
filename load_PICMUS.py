# File:       load_PICMUS.py
# Author:     Dongwoon Hyun (dongwoon.hyun@stanford.edu)
# Created on: 2020-03-12

import h5py
import numpy as np
from scipy.signal import hilbert


class PlaneWaveData:
    def __init__(self, database_path, acq, target, dtype):
        assert any([acq == a for a in ["simulation", "experiments"]])
        assert any([target == t for t in ["contrast_speckle", "resolution_distorsion"]])
        assert any([dtype == d for d in ["rf", "iq"]])
        fname = "%s/%s/%s/%s_%s_dataset_%s.hdf5" % (
            database_path,
            acq,
            target,
            target,
            acq[:4],
            dtype,
        )

        # Load PICMUS dataset
        f = h5py.File(fname, "r")["US"]["US_DATASET0000"]
        self.idata = np.array(f["data"]["real"], dtype="float32")
        self.qdata = np.array(f["data"]["imag"], dtype="float32")
        self.angles = np.array(f["angles"])
        self.fc = 5208000.0  # np.array(f["modulation_frequency"]).item()
        self.fs = np.array(f["sampling_frequency"]).item()
        self.c = np.array(f["sound_speed"]).item()
        self.time_zero = np.array(f["initial_time"])
        self.ele_pos = np.array(f["probe_geometry"]).T
        self.fdemod = self.fc if dtype == "iq" else 0

        if dtype == "rf":
            iqdata = hilbert(self.idata, axis=-1)
            self.qdata = np.imag(iqdata)

        # Make sure that time_zero is an array of size [nangles]
        if self.time_zero.size == 1:
            self.time_zero = np.ones_like(self.angles) * self.time_zero
