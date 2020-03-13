# CUBDL - Challenge on Ultrasound Beamforming with Deep Learning

## Description

This repository is for the IEEE 2020 International Ultrasonics Symposium Challenge on Ultrasound Beamforming with Deep Learning. Example python starter code is provided to demonstrate the input data format and the metrics.

## Input data format

During testing, we will provide the user with an object describing the acquisition, as well as a pixel grid to be reconstructed.

### Plane wave acquisition description

A description of the plane wave acquisition will be provided as a `PlaneWaveData` object `P` with the following attributes:

- `idata` - A 3D numpy array of the real component of the raw channel signals for a plane wave acquisition. The dimensions can be described as

  - Python: `nangles, nelements, nsamples = idata.shape`
  - MATLAB: `[nsamples, nelements, nangles] = size(idata);`

  The MATLAB expression is provided only as reference. Testing will be performed in python. Note that `idata` can be either RF-modulated or demodulated (baseband), as determined by the `fdemod` parameter listed below.
- `qdata` - A 3D numpy array for the quadrature component of `idata`.
- `angles` - A 1D numpy array of shape `[nangles,]` of the transmitted plane wave angles in radians.
- `fc` - The center frequency of the transmitted RF waveform.
- `fs` - The sampling frequency of the acquisition.
- `c` - The nominal speed of sound to be used for reconstruction.
- `time_zero` - A 1D numpy array of shape `[nangles,]` listing the time considered to be **time zero** for the purposes of image reconstruction on a per-transmission basis.
- `ele_pos` - A 2D numpy array of shape `[nelements, 3]` describing the azimuthal, elevation, and depth coordinates of each element.
- `fdemod` - The frequency of demodulation applied to the raw data. It is often desirable to demodulate the data prior to focusing to lower the Nyquist frequency (i.e., require sufficient sampling with respect to the bandwidth, rather than the center frequency plus bandwidth). In the case that no demodulation is applied (i.e., the data is still modulated), `fdemod = 0`.

### Beamforming pixel grid

The desired pixel grid to be reconstructed is specified as

- `grid` - A 3D numpy array with dimensions

  - Python: `ncols, nrows, 3 = grid.shape`
  - MATLAB: `[xyz, nrows, ncols] = size(grid);`.

### Example delay-and-sum code

A simple example of delay-and-sum beamforming is provided in [das_torch.py](das_torch.py).

### Metrics

Coming soon: Metrics will be provided in [metrics.py](metrics.py).

## Example code prerequisites

The example code uses numpy and PyTorch to perform delay-and-sum beamforming. PyTorch can be installed to execute on a CUDA-enabled GPU by creating an anaconda environment with

```shell
conda create -n cubdl python=3 pytorch torchvision cudatoolkit=10.1 -c pytorch
```
