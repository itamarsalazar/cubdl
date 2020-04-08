# File:       das_torch.py
# Author:     Dongwoon Hyun (dongwoon.hyun@stanford.edu)
# Created on: 2020-03-09
import torch
from torch.nn.functional import grid_sample
from tqdm import tqdm

PI = 3.14159265359


class DAS_PW(torch.nn.Module):
    """ PyTorch implementation of DAS plane wave beamforming.

    This class implements DAS plane wave beamforming as a neural network via a PyTorch
    nn.Module. Subclasses derived from this class can choose to make certain parameters
    trainable. All components can be turned into trainable parameters.
    """

    def __init__(
        self,
        P,
        grid,
        ang_list=None,
        ele_list=None,
        rxfnum=2,
        dtype=torch.float,
        device=torch.device("cuda:0"),
    ):
        """ Initialization method for DAS_PW.

        All inputs are specified in SI units, and stored in self as PyTorch tensors.
        INPUTS
        P           A PlaneWaveData object that describes the acquisition
        grid        A [ncols, nrows, 3] numpy array of the reconstruction grid
        ang_list    A list of the angles to use in the reconstruction
        ele_list    A list of the elements to use in the reconstruction
        rxfnum      The f-number to use for receive apodization
        dtype       The torch Tensor datatype (defaults to torch.float)
        device      The torch Tensor device (defaults to GPU execution)
        """
        super().__init__()
        # If no angle or element list is provided, delay-and-sum all
        if ang_list is None:
            ang_list = range(P.angles.shape[0])
        elif not hasattr(ang_list, "__getitem__"):
            ang_list = [ang_list]
        if ele_list is None:
            ele_list = range(P.ele_pos.shape[0])
        elif not hasattr(ele_list, "__getitem__"):
            ele_list = [ele_list]

        # Convert plane wave data to tensors
        self.angles = torch.tensor(P.angles, dtype=dtype, device=device)
        self.ele_pos = torch.tensor(P.ele_pos, dtype=dtype, device=device)
        self.fc = torch.tensor(P.fc, dtype=dtype, device=device)
        self.fs = torch.tensor(P.fs, dtype=dtype, device=device)
        self.fdemod = torch.tensor(P.fdemod, dtype=dtype, device=device)
        self.c = torch.tensor(P.c, dtype=dtype, device=device)
        self.time_zero = torch.tensor(P.time_zero, dtype=dtype, device=device)

        # Convert grid to tensor
        self.grid = torch.tensor(grid, dtype=dtype, device=device).view(-1, 3)
        self.out_shape = grid.shape[:-1]

        # Store other information as well
        self.ang_list = torch.tensor(ang_list, dtype=torch.long, device=device)
        self.ele_list = torch.tensor(ele_list, dtype=torch.long, device=device)
        self.dtype = dtype
        self.device = device

    def forward(self, x):
        """ Forward pass for DAS_PW neural network.
        
        """
        idata, qdata = x
        dtype, device = self.dtype, self.device

        # Compute delays in meters
        nangles = len(self.ang_list)
        nelems = len(self.ele_list)
        npixels = self.grid.shape[0]
        xlims = (self.ele_pos[0, 0], self.ele_pos[-1, 0])  # Aperture width
        txdel = torch.zeros((nangles, npixels), dtype=dtype, device=device)
        rxdel = torch.zeros((nelems, npixels), dtype=dtype, device=device)
        txapo = torch.ones((nangles, npixels), dtype=dtype, device=device)
        rxapo = torch.ones((nelems, npixels), dtype=dtype, device=device)
        for i, tx in enumerate(self.ang_list):
            txdel[i] = delay_plane(self.grid, self.angles[[tx]], xlims)
            txdel[i] += self.time_zero[tx] * self.c
        for j, rx in enumerate(self.ele_list):
            rxdel[j] = delay_focus(self.grid, self.ele_pos[[rx]])
        # Convert to samples
        txdel *= self.fs / self.c
        rxdel *= self.fs / self.c

        # Initialize the output array
        idas = torch.zeros(npixels, dtype=self.dtype, device=self.device)
        qdas = torch.zeros(npixels, dtype=self.dtype, device=self.device)
        # Loop over angles and elements
        for tx, td, ta in tqdm(zip(self.ang_list, txdel, txapo), total=nangles):
            for rx, rd, ra in zip(self.ele_list, rxdel, rxapo):
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
                if self.fdemod != 0:
                    tshift = delays.view(-1) / self.fs - self.grid[:, 2] * 2 / self.c
                    theta = 2 * 3.14159265359 * self.fdemod * tshift
                    itmp, qtmp = _complex_rotate(itmp, qtmp, theta)
                # Apply apodization, reshape, and add to running sum
                apods = ta * ra
                idas += itmp * apods
                qdas += qtmp * apods

        # Finally, restore the original pixel grid shape and convert to numpy array
        idas = idas.view(self.out_shape)
        qdas = qdas.view(self.out_shape)
        return idas, qdas


## Simple phase rotation of I and Q component by complex angle theta
def _complex_rotate(I, Q, theta):
    Ir = I * torch.cos(theta) - Q * torch.sin(theta)
    Qr = Q * torch.cos(theta) + I * torch.sin(theta)
    return Ir, Qr


## Compute distance to user-defined pixels from elements
# Expects all inputs to be torch tensors specified in SI units.
# INPUTS
#   grid    Pixel positions in x,y,z    [npixels, 3]
#   ele_pos Element positions in x,y,z  [nelems, 3]
# OUTPUTS
#   dist    Distance from each pixel to each element [nelems, npixels]
def delay_focus(grid, ele_pos):
    # Get norm of distance vector between elements and pixels via broadcasting
    dist = torch.norm(grid - ele_pos.unsqueeze(0), dim=-1)
    # Output has shape [nelems, npixels]
    return dist


## Compute distance to user-defined pixels for plane waves
# The distance is computed as x * sin(apod) + z * cos(apod) + w/2 * sin(|apod|)
# Expects all inputs to be torch tensors specified in SI units.
# INPUTS
#   grid    Pixel positions in x,y,z    [npixels, 3]
#   angles  Plane wave angles (radians) [nangles]
#   xlims   Azimuthal limits of the aperture [2]
# OUTPUTS
#   dist    Distance from each pixel to each element [nelems, npixels]
def delay_plane(grid, angles, xlims):
    # Compute width of full aperture (to determine angular offset)
    w = (xlims[1] - xlims[0]).abs()
    # Use broadcasting to simplify computations
    x = grid[:, 0].unsqueeze(0)
    z = grid[:, 2].unsqueeze(0)
    # For each element, compute distance to pixels
    dist = x * torch.sin(angles) + z * torch.cos(angles)
    # Output has shape [nangles, npixels]
    return dist


## Compute rect apodization to user-defined pixels for desired f-number
# Expects all inputs to be torch tensors specified in SI units.
# INPUTS
#   grid        Pixel positions in x,y,z        [npixels, 3]
#   ele_pos     Element positions in x,y,z      [nelems, 3]
#   fnum        Desired f-number                scalar
#   min_width   Minimum width to retain         scalar
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
#   grid    Pixel positions in x,y,z            [npixels, 3]
#   angles  Plane wave angles (radians)         [nangles]
#   xlims   Azimuthal limits of the aperture    [2]
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
