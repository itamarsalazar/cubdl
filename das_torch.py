# File:       das_torch.py
# Author:     Dongwoon Hyun (dongwoon.hyun@stanford.edu)
# Created on: 2020-03-09
import torch
from torch.nn.functional import grid_sample
from tqdm import tqdm

PI = 3.14159265359

## Simple phase rotation of I and Q component by complex angle theta
def _complex_rotate(I, Q, theta):
    Ir = I * torch.cos(theta) - Q * torch.sin(theta)
    Qr = Q * torch.cos(theta) + I * torch.sin(theta)
    return Ir, Qr


## PyTorch implementation of DAS
# Expects all inputs to be torch tensors specified in SI units.
# INPUTS
#   idata       Real component of data  [nxmits, nelems, nsamps]
#   qdata       Imag component of data  [nxmits, nelems, nsamps]
#   txdel       Transmit delay profile  [nxmits, npixels]
#   rxdel       Receive delay profile   [nelems, npixels]
#   txapo       Transmit apodization    [nxmits, npixels]
#   rxapo       Receive apodization     [nxmits, npixels]
#   outshape    Desired shape of output [npixel_dims]
#   fdemod      Demodulation frequency  scalar, 0 if data is RF-modulated
# OUTPUTS
#   idas    Real delay-and-summed output    [npixels]
#   qdas    Imag delay-and-summed output    [npixels]
def DAS_torch(
    P,
    grid,
    ang_list=None,
    ele_list=None,
    rxfnum=2,
    dtype=torch.float,
    device=torch.device("cuda:0"),
):

    # If no angle or element list is provided, delay-and-sum all
    if ang_list is None:
        ang_list = range(P.angles.shape[0])
    elif not isinstance(ang_list, list):
        ang_list = [ang_list]
    if ele_list is None:
        ele_list = range(P.ele_pos.shape[0])
    elif not isinstance(ele_list, list):
        ele_list = [ele_list]

    # For speed, put everything on the device as torch tensors
    idata = torch.tensor(P.idata, dtype=dtype, device=device)
    qdata = torch.tensor(P.qdata, dtype=dtype, device=device)
    angles = torch.tensor(P.angles, dtype=dtype, device=device)
    ele_pos = torch.tensor(P.ele_pos, dtype=dtype, device=device)
    grid = torch.tensor(grid, dtype=dtype, device=device)

    # Store pixel grid shape and reshape into [npixels, 3]
    out_shape = grid.shape[:-1]  # Keep output grid shape for later use
    grid = grid.view(-1, 3)  # Reshape
    npixels = grid.shape[0]

    # Precompute the delays and apodizations for faster processing
    nangles = len(ang_list)
    nelems = len(ele_list)
    xlims = [ele_pos[0, 0], ele_pos[-1, 0]]
    txdel = torch.zeros((nangles, npixels), dtype=dtype, device=device)
    txapo = torch.zeros((nangles, npixels), dtype=dtype, device=device)
    rxdel = torch.zeros((nelems, npixels), dtype=dtype, device=device)
    rxapo = torch.zeros((nelems, npixels), dtype=dtype, device=device)
    time_zero = torch.zeros((nangles, 1), dtype=dtype, device=device)
    for i, tx in enumerate(ang_list):
        txdel[i] = delay_plane(grid, angles[[tx]], xlims) * P.fs / P.c
        txapo[i] = apod_plane(grid, angles[[tx]], xlims) * 0 + 1
        time_zero[i] = torch.tensor(P.time_zero[[tx]], dtype=dtype, device=device)
    for j, rx in enumerate(ele_list):
        rxdel[j] = delay_focus(grid, ele_pos[[rx]]) * P.fs / P.c
        rxapo[j] = apod_focus(grid, ele_pos[[rx]], fnum=rxfnum)
    txdel += time_zero * P.fs

    # Initialize the output array
    # idas = torch.zeros(out_shape, dtype=dtype, device=device)
    # qdas = torch.zeros(out_shape, dtype=dtype, device=device)
    idas = torch.zeros(npixels, dtype=dtype, device=device)
    qdas = torch.zeros(npixels, dtype=dtype, device=device)
    # Loop over angles and elements
    for i, (tx, td, ta) in tqdm(enumerate(zip(ang_list, txdel, txapo)), total=nangles):
        for j, (rx, rd, ra) in enumerate(zip(ele_list, rxdel, rxapo)):
            # Grab data from i-th Tx, j-th Rx
            I = idata[tx, rx].view(1, 1, 1, -1)
            Q = qdata[tx, rx].view(1, 1, 1, -1)
            # Convert delays to be used with grid_sample
            delays = td + rd
            d_gs = (delays.view(1, 1, -1, 1) * 2 + 1) / idata.shape[-1] - 1
            d_gs = torch.cat((d_gs, 0 * d_gs), axis=-1)
            # Interpolate using grid_sample and vectorize using view(-1)
            itmp = grid_sample(I, d_gs, align_corners=False).view(-1)
            qtmp = grid_sample(Q, d_gs, align_corners=False).view(-1)
            # Apply phase-rotation if focusing demodulated data
            if P.fdemod != 0:
                tshift = delays.view(-1) / P.fs - grid[:, 2] * 2 / P.c
                theta = 2 * PI * P.fdemod * tshift
                itmp, qtmp = _complex_rotate(itmp, qtmp, theta)
            # Apply apodization, reshape, and add to running sum
            apods = ta * ra
            idas += itmp * apods
            qdas += qtmp * apods

    # Finally, restore the original pixel grid shape and convert to numpy array
    idas = idas.view(out_shape).cpu().numpy()
    qdas = qdas.view(out_shape).cpu().numpy()
    return idas, qdas


## Compute distance to user-defined pixels from elements
# Expects all inputs to be torch tensors specified in SI units.
# INPUTS
#   grid Pixel positions in x,y,z    [npixels, 3]
#   ele_pos Element positions in x,y,z  [nelems, 3]
# OUTPUTS
#   dist    Distance from each pixel to each element [nelems, npixels]
def delay_focus(grid, ele_pos):
    # Get norm of distance vector between elements and pixels via broadcasting
    dist = torch.norm(grid.unsqueeze(0) - ele_pos.unsqueeze(1), dim=-1)
    # Output has shape [nelems, npixels]
    return dist


## Compute distance to user-defined pixels for plane waves
# The distance is computed as x * sin(apod) + z * cos(apod) + w/2 * sin(|apod|)
# Expects all inputs to be torch tensors specified in SI units.
# INPUTS
#   grid Pixel positions in x,y,z    [npixels, 3]
#   ele_pos Element positions in x,y,z  [nelems, 3]
#   angles  Plane wave angles (radians) [nangles]
# OUTPUTS
#   dist    Distance from each pixel to each element [nelems, npixels]
def delay_plane(grid, angles, xlims):
    # Compute width of full aperture (to determine angular offset)
    w = (xlims[1] - xlims[0]).abs()
    # Use broadcasting to simplify computations
    a = angles.unsqueeze(1)
    x = grid[:, 0].unsqueeze(0)
    z = grid[:, 2].unsqueeze(0)
    # For each element, compute distance to pixels
    dist = x * torch.sin(a) + z * torch.cos(a)
    # Output has shape [nangles, npixels]
    return dist


## Compute rect apodization to user-defined pixels for desired f-number
# Expects all inputs to be torch tensors specified in SI units.
# INPUTS
#   grid Pixel positions in x,y,z                [npixels, 3]
#   ele_pos Element positions in x,y,z              [nelems, 3]
#   fnum    Desired f-number                        scalar
#   min_ele Minimum number of elements to retain    scalar
# OUTPUTS
#   apod    Apodization for each pixel to each element  [nelems, npixels]
def apod_focus(grid, ele_pos, fnum=1, min_width=1e-3):
    # Get vector between elements and pixels via broadcasting
    ppos = grid.unsqueeze(0)
    epos = ele_pos.unsqueeze(1)
    v = ppos - epos
    # Select (ele,pix) pairs whose effective fnum is greater than fnum
    mask = torch.abs(v[:, :, 2] / v[:, :, 0]) > fnum
    mask = mask | (torch.abs(v[:, :, 0]) <= min_width)
    # Also account for edges of aperture
    mask = mask | ((v[:, :, 0] >= min_width) & (ppos[:, :, 0] <= ele_pos[0, 0]))
    mask = mask | ((v[:, :, 0] <= -min_width) & (ppos[:, :, 0] >= ele_pos[-1, 0]))
    # Convert to float and normalize across elements (i.e., delay-and-"average")
    apod = mask.float()
    apod /= torch.sum(apod, 1, keepdim=True)
    # Output has shape [nelems, npixels]
    return apod


## Compute rect apodization to user-defined pixels for desired f-number
# Retain only pixels that lie within the aperture projected along the transmit angle.
# Expects all inputs to be torch tensors specified in SI units.
# INPUTS
#   grid Pixel positions in x,y,z        [npixels, 3]
#   ele_pos Element positions in x,y,z      [nelems, 3]
#   angles  Plane wave angles (radians)     [nangles]
# OUTPUTS
#   apod    Apodization for each angle to each element  [nangles, npixels]
def apod_plane(grid, angles, xlims):
    pix = grid.unsqueeze(0)
    ang = angles.unsqueeze(1)
    # Project pixels back to aperture along the defined angles
    x_proj = pix[:, :, 0] - pix[:, :, 2] * torch.tan(ang)
    # Select only pixels whose projection lie within the aperture, with fudge factor
    mask = (x_proj >= xlims[0] * 1.2) & (x_proj <= xlims[1] * 1.2)
    # Convert to float and normalize across angles (i.e., delay-and-"average")
    apod = mask.float()
    apod /= torch.sum(apod, 1, keepdim=True)
    # Output has shape [nangles, npixels]
    return apod
