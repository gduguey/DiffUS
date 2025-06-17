import time

from tqdm import tqdm
import torch
import torch.nn.functional as F
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as ss
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d

from matplotlib.widgets import Slider
# from scipy.signal import ricker

import warnings

class UltrasoundRenderer:
    def __init__(self, num_samples: int, attenuation_coeff: float = 0.5):
        """
        num_samples: how many points to sample along each ray
        attenuation_coeff: controls exponential decay of echoes with depth
        """
        self.num_samples = num_samples
        self.attenuation_coeff = attenuation_coeff

    @staticmethod
    def compute_reflection_coeff(Z1: torch.Tensor, Z2: torch.Tensor) -> torch.Tensor:
        """
        Compute power reflection coefficient between two impedances.
        R = ((Z2 - Z1)/(Z2 + Z1))**2
        """
        return (Z2 - Z1) / (Z1+Z2)

    def simulate_rays(self,
        volume: torch.Tensor, 
        source: torch.Tensor, 
        directions: torch.Tensor, 
        num_samples: int = 0,
        MRI:bool=False,
        start=0) -> torch.Tensor:
        """
        Simulates a single ray tracing through a 3D volume using batched grid_sample.
        
        Args:
            volume: (D, H, W) Tensor of acoustic properties (e.g., normalized impedance)
            source: (3,) Tensor for the starting point of the ray
            direction: (n_rays,3,) Tensor for the ray directions (must be a unit vector) If a unique direction is given, it can be of shape (3,)
            num_samples: int, number of steps along the ray
            
        Returns:
            R: (num_samples-1) Tensor of reflection coefficients along the ray
        """
        if num_samples == 0:
            num_samples = self.num_samples
                
        impedances =  self.trace_ray(
            volume=volume, 
            source=source, 
            directions=directions, 
            num_samples=num_samples)
        if impedances.ndim == 1:
            impedances = impedances.unsqueeze(0)
        Z1 = impedances[:, :-1]  # (num_samples-1)
        Z2 = impedances[:, 1:]   # (num_samples-1)

        R = self.compute_reflection_coeff(Z1, Z2)
        if MRI:
            return Z1
        return R.squeeze(0) 

    def simulate_frame(self,
                       volume: torch.Tensor,
                       source: torch.Tensor,
                       directions: torch.Tensor) -> torch.Tensor:
        """
        # DEPRECATED
        Simulate a full ultrasound frame.
        - directions: (N_rays, 3) each row a unit vector
        Returns: (N_rays, num_samples - 1) intensity map
        """
        warnings.warn("This function is deprecated. Use simulate_rays instead, it can parse a batch of directions.")
        return torch.stack([
            self.simulate_rays(volume, source, d)
            for d in directions
        ], dim=0)
    
    @staticmethod
    def trace_ray(
                volume: torch.Tensor, 
                source: torch.Tensor, 
                directions: torch.Tensor, 
                num_samples: int) -> torch.Tensor:
        """
        Simulates ray tracing through a 3D volume using batched grid_sample.
        
        Args:
            volume: (H, W, D) Tensor of acoustic properties (e.g., normalized impedance) oriented such that [:,:,i] is an axial slice at depth i
            source: (3,) Tensor for the starting point of rays
            directions: (N_rays, 3) Tensor of ray directions (must be unit vectors)
            num_samples: int, number of steps along each ray
            
        Returns:
            R: (N_rays, num_samples-1) Tensor of reflection coefficients along each ray
        """
        n_rays = directions.shape[0]
        # housekeeping
        if directions.ndim == 1:
            directions = directions.unsqueeze(0)
        
        # Z = (volume - volume.min()) / (volume.max() - volume.min())
        # Z = Z * (1.7e6 - 1.4e6) + 1.4e6
        Z = volume  # assume volume is already normalized impedance
        
        H, W, D = volume.shape
        
        # Prepare steps and directions
        steps = torch.arange(0, num_samples, dtype=torch.float32, device=volume.device).view(1, -1, 1)  # (1, num_samples, 1)
        directions = directions.unsqueeze(1)  # (N_rays, 1, 3)
        assert directions.shape[1] == 1, "Directions must be a unit vector for each ray."
        print("[INFO] Tracing rays with source:", source, "and directions shape:", directions.shape)
        # Trace points
        points = source + steps * directions 
        if False:
            # Compute variance across each dimension
            var_x = points[..., 0].var().item()
            var_y = points[..., 1].var().item()
            var_z = points[..., 2].var().item()
            variances = [var_x, var_y, var_z]
            drop_axis = variances.index(min(variances))  # axis with least variance

            # Set axis labels for clarity
            axis_names = ['x', 'y', 'z']
            keep_axes = [i for i in range(3) if i != drop_axis]
            ax0, ax1 = keep_axes

            # Extract the 2D projection of points
            proj0 = points[:, :, ax0].flatten().cpu()
            proj1 = points[:, :, ax1].flatten().cpu()

            # Choose the corresponding slice of the volume
            vol_np = volume.cpu().numpy()
            slice_idx = int(points[0, 0, drop_axis].item())  # round to nearest index
            slice_idx = max(0, min(slice_idx, volume.shape[drop_axis] - 1))

            if drop_axis == 0:
                slice_img = vol_np[slice_idx, :, :]
            elif drop_axis == 1:
                slice_img = vol_np[:, slice_idx, :]
            else:
                slice_img = vol_np[:, :, slice_idx]

            # Plot
            plt.figure(figsize=(6, 6))
            plt.imshow(slice_img, cmap='gray', alpha=0.5, origin='lower')
            plt.scatter(proj0, proj1, s=1, c='red', alpha=0.5)
            plt.title(f"Projection in {axis_names[ax0]}-{axis_names[ax1]} plane")
            plt.xlabel(axis_names[ax0])
            plt.ylabel(axis_names[ax1])
            plt.show()

        # Ensure Z is in (D, H, W) order
        Z = Z.permute(2, 0, 1).unsqueeze(0).unsqueeze(0).contiguous().float()  # (1, 1, D, H, W)
        grid = torch.empty_like(points, dtype=torch.float32, device=volume.device)
        grid[..., 0] = 2 * (points[..., 0] / (W - 1)) - 1  # x
        grid[..., 1] = 2 * (points[..., 1] / (H - 1)) - 1  # y
        grid[..., 2] = 2 * (points[..., 2] / (D - 1)) - 1  # z
        grid = grid.view(1, n_rays, num_samples, 1, 3)
        if False:  # DEBUG
            print("[INFO] Grid shape:", grid.shape, "Z shape:", Z.shape)
        # Sampling
        ray_values = F.grid_sample(
            Z, grid, mode='bilinear', padding_mode='border', align_corners=True
        ).squeeze()
        return ray_values
    
    def trace_rays(self, volume, sources, directions, num_samples):
        """
        Trace multiple rays from a set of sources in the same direction.

        Args:
            volume (torch.Tensor): (D, H, W)
            sources (torch.Tensor): (N, 3) source points
            direction (tuple or torch.Tensor): (dx, dy, dz)
            num_samples (int): Number of steps per ray

        Returns:
            torch.Tensor: (N, num_samples) intensity values for each ray
        """
        all_profiles = []
        for src in sources:
            profile = self.trace_ray(volume, src, directions, num_samples)
            all_profiles.append(profile)
        return torch.stack(all_profiles)  # (N, num_samples)

    def plot_beam_frame(
        self,
        volume: torch.Tensor,
        source: torch.Tensor,
        directions: torch.Tensor,
        angle: float = 45.0,
        plot: bool = True,
        artifacts: bool = False,
        ax: plt.Axes = None,
        cmap: bool = None,
        std_radial: float = 0.000002,
        std_local: float = 0.0000005,
        max_sigma: float = 2.0,
        alpha: float = 1.5,
        start: float = 0,
        **kwargs  # for future extensibility, e.g. MRI=False
    ):
        """
        Simulates rays and plots the resulting ultrasound fan frame.
        
        Args:
            volume: (D, H, W) Tensor, ultrasound volume (acoustic impedance)
            source: (3,) Tensor, starting position of rays
            directions: (N_rays, 3) Tensor of ray directions (unit vectors)
            angle: fan angle (degrees), used for plotting geometry
        """
        # 1. Simulate reflection coefficients
        R = self.simulate_rays(
            volume=volume,
            source=source,
            directions=directions,
            MRI=False,
        ) # This samples the volume 
        device = R.device
        # Start index handling
        if type(start) is float:
            start = int(start * self.num_samples)
        if type(start) is int:
            start = max(0, start)  # ensure non-negative index
        if start > 0:
            R = R[:, start:]
            print("[INFO] Starting from sample index:", start, "(for instance, to skip bones)")
        
        # DEPRECATED THIS: # processed_output = propagate_full_rays_batched(R)  # (N_rays, num_samples-1, 2*(num_samples-1))

        processed_output = compute_gaussian_pulse(R, length=20, sigma=2) # (N_rays, num_samples-1)s
        # processed_output = R
        
        # Attenuation model
        attenuation_coeff = 0.001
        depths = torch.arange(processed_output.shape[1], device=device).float()
        attenuation = torch.exp(-attenuation_coeff * depths)  # shape (num_samples,)
        processed_output = processed_output * attenuation[None, :]  # shape (N_rays, num_samples)
        
        processed_output_torch = processed_output.clone()        
        
        # reconstruct the start
        if artifacts:
            # speckle
            processed_output = add_speckle_arcs_np(processed_output, 
                                            std_radial=std_radial,
                                            std_local=std_local)
            # lateral blur
            processed_output = add_depth_dependent_lateral_blur_np(processed_output,
    max_sigma=max_sigma)
            # sharpen
            processed_output = sharpen_np(processed_output, alpha=alpha)   
        
        if start > 0:
            # processed_output = np.pad(processed_output, ((0, 0), (start, 0)), mode='constant', constant_values=0)
            processed_output = F.pad(processed_output, pad=(start, 0, 0, 0), mode='constant', value=0)
            print("[INFO] Padded output to start from sample index:", start, processed_output.shape)
        
        # 2. Convert to numpy
        n_rays, n_samples = processed_output.shape

        # # 3. Compute ray geometry
        # source_2d = torch.tensor([128.0, 0.0])  # (x, z), assuming 2D fan centered at (128, 0)
        # thetas = torch.deg2rad(torch.linspace(-angle, angle, n_rays))  # shape (n_rays,)
        ray_len = n_samples  # number of points along each ray

        # Vectorized generation of all points
        steps = np.arange(ray_len)  # (n_samples,)

        # # Ray directions in 2D
        # directions_xz = torch.stack([
        #     torch.sin(thetas),  # (n_rays,)
        #     torch.cos(thetas)
        # ], dim=1)  # (n_rays, 2)

        # Modifs Gabi push pour Noé
        source_2d = source[:2]
        directions_xz = directions[:, :2]  # Use the first two components for 2D projection
        # Expand dimensions
        steps = torch.arange(n_samples, dtype=torch.float32, device=device).view(1, n_samples, 1)  # (1, n_samples, 1)
        directions_xz = directions_xz.view(n_rays, 1, 2)                             # (n_rays, 1, 2)

        points = source_2d.view(1, 1, 2) + steps * directions_xz  # (n_rays, n_samples, 2)

        points = points[:, start:, :]  # slice to start index if needed
        processed_output = processed_output[:, start:]  # slice to start index if needed
        x_coords = points[..., 0].flatten()
        z_coords = points[..., 1].flatten()
        intensities = processed_output.flatten()
        if not plot:
            return x_coords, z_coords, intensities
        
        # -- 4. Plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        
        ax.set_facecolor("black")
        ## SET CUSTOM VMIN VMAX BASED ON HISTOGRAM

        threshold_min = 0
        threshold_max = intensities.max()

        if threshold_max < 0:
            warnings.warn("Intensity values contain negative values. Displaying full range.")
            threshold_min = intensities.min()

        print(f"Intensity range: {threshold_min:.4f} to {threshold_max:.4f}")
        if ax is None:
            fig, ax = plt.subplots()
            plt.subplots_adjust(left=0.1, bottom=0.25)

        # Initial plot
        scatter = ax.scatter(
            x_coords, z_coords, c=intensities,
            cmap=cmap if cmap is not None else 'gray',
            s=2, vmin=threshold_min, vmax=threshold_max * 1.5
        )
        ax.set_aspect('equal')
        ax.set_title("Fan-shaped Ultrasound Frame")

        ax.set_xticks([])
        ax.set_yticks([])

        # 4. 
        return processed_output_torch, scatter

    @staticmethod
    def plot_frame(frame: torch.Tensor, ax: plt.Axes = None):
        """
        Display the simulated US frame.
        - frame: (N_rays, depth) intensity map
        """
        if ax is None:
            plt.figure(figsize=(6, 6))
            ax = plt.gca()
        
        # transpose so depth goes downwards
        frame_np = frame.T.cpu().numpy()
        ax.imshow(frame_np, cmap='gray', aspect='auto', vmin=frame_np.min(), vmax=frame_np.max())
        ax.set_xlabel('Ray index')
        ax.set_ylabel('Depth sample')
        ax.set_title('Input Volume Slice')
        return ax

    def plot_sector(self,
                    frame: torch.Tensor,
                    angles: torch.Tensor,
                    spacing: float = 1.0):
        """
        Plot the US frame in true sector geometry.
        - frame: (N_rays, depth) intensity map
        - angles: (N_rays,) ray angles in radians
        - spacing: real-world depth per sample (e.g. mm per voxel)
        """
        N_rays, depth = frame.shape
        depths = torch.arange(0, depth, dtype=frame.dtype) * spacing

        # gather (x,z) coords + intensity
        xs, zs, vals = [], [], []
        for i, theta in enumerate(angles):
            sin, cos = torch.sin(theta), torch.cos(theta)
            for j, r in enumerate(depths):
                xs.append((r * sin).item())
                zs.append((r * cos).item())
                vals.append(frame[i, j].item())

        # scatter
        plt.figure(figsize=(6, 6))
        plt.scatter(xs, zs, c=vals, s=1, cmap='gray', vmin=min(vals), vmax=max(vals))
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.invert_yaxis() 
        plt.xlabel('x (lateral)')
        plt.ylabel('z (depth)')
        plt.title('Sector-shaped US image')
        plt.colorbar(label='Echo intensity')
        plt.show()
   

    
    def plot_sector_bmode(self,
                        bmode: np.ndarray,
                        angles: np.ndarray,
                        spacing: float = 1.0):
        """
        DEPR?
        Plot the B-mode image in true sector geometry.
        - bmode: (N_rays, depth) intensity map (NumPy array, normalized to [0, 1])
        - angles: (N_rays,) ray angles in radians
        - spacing: real-world depth per sample (e.g., mm per voxel)
        """
        N_rays, depth = bmode.shape
        depths = np.arange(0, depth) * spacing

        # Gather (x, z) coordinates and intensity values
        xs, zs, vals = [], [], []
        for i, theta in enumerate(angles):
            sin, cos = np.sin(theta), np.cos(theta)
            for j, r in enumerate(depths):
                xs.append(r * sin)
                zs.append(r * cos)
                vals.append(bmode[i, j])

        # Scatter plot
        plt.figure(figsize=(6, 6))
        plt.scatter(xs, zs, c=vals, s=1, cmap='gray', vmin=min(vals), vmax=max(vals))
        ax = plt.gca()
        ax.set_aspect('equal')
        # ax.invert_yaxis()
        plt.xlabel('x (lateral)')
        plt.ylabel('z (depth)')
        plt.title('Sector-shaped B-mode Ultrasound Image')
        plt.colorbar(label='Normalized intensity')
        plt.show()

    
# UTILITIES

def prop_single_ray(refLR: torch.Tensor, traLR: torch.Tensor = None, traRL: torch.Tensor = None) -> torch.Tensor:
    """
    Solve the 2(N+1)×2(N+1) system for *each* ray in the batch.

    Parameters
    ----------
    refLR : (B, N)  reflection coeffs for incidence from the left

    Returns
    -------
    w : (B, 2*(N+1)) laid out as [g0, d0, g1, d1, …, gN, dN]
    """
    B, N = refLR.shape
    traLR = 1 + refLR              # t_il
    traRL = 1 - refLR              # t_ir
    refRL = -refLR                 # r_rl  (OK only if impedances equal)

    size = 2 * (N + 1)
    A = torch.zeros((B, size, size), dtype=refLR.dtype, device=refLR.device)
    b = torch.zeros((B, size),      dtype=refLR.dtype, device=refLR.device)

    # boundary conditions: g0 = 1 , d_{N+1}=0
    b[:, 0] = 1
    A[:, 0, 0]   = 1
    A[:, -1, -1] = 1

    for i in range(N):
        gi,  di   = 2 * i,     2 * i + 1          # positions inside the vector
        gip1, dip1 = 2 * (i + 1), 2 * (i + 1) + 1

        # Eq. 1 :  g_{i+1} - t_il g_i - r_lr d_{i+1} = 0
        A[:, gip1, gi]   = -traLR[:, i]           # –t_il g_i
        A[:, gip1, dip1] = -refLR[:, i]           # –r_lr d_{i+1}
        A[:, gip1, gip1] =  1                     # +g_{i+1}

        # Eq. 2 :  d_i - r_rl g_i - t_ir d_{i+1} = 0
        A[:, di, gi]     = -refRL[:, i]           # –r_rl g_i
        A[:, di, dip1]   = -traRL[:, i]           # –t_ir d_{i+1}
        A[:, di, di]     =  1                     # +d_i

    w = torch.linalg.solve(A, b)                  # (B, 2*(N+1))
    # print(w)
    return w

def propagate_full_rays_batched(refLR: torch.Tensor) -> torch.Tensor:
    """
    For each truncation depth i = 0..N compute the surface-return amplitude d0^{(i)}.

    Parameters
    ----------
    refLR : (B, N)  reflection coeffs for incidence from the left

    Returns
    -------
    d0_per_depth : (B, N+1)  with d0^{(0)}, d0^{(1)}, …, d0^{(N)}
    """
    # print(refLR.shape)
    B, N = refLR.shape
    d0_per_depth = []
    total = []
    for i in range(N + 1):
        w = prop_single_ray(refLR[:, :i])        # solve the truncated system
        d0_per_depth.append(w[:, 1])       # column 1 is d0
        total.append(w)
        
    # return total
    stacked_d0 =  torch.stack(d0_per_depth, dim=1)
    stacked_d0 = torch.cumsum(stacked_d0, dim=1)  # cumulative sum along the depth axis
    return stacked_d0


def compute_echo_traces(refLR: torch.Tensor, spacing: float = 1.0, c: float = 1.54e3) -> torch.Tensor:
    """
    Compute the echo traces for each ray in the batch.

    Parameters
    ----------
    refLR : (B, N)  reflection coeffs for incidence from the left
    spacing : float, spacing between samples in mm
    c : float, speed of sound in m/s

    Returns
    -------
    traces : (B, N)  echo traces for each ray
    """
    d0_series = propagate_full_rays_batched(refLR)  # (B, N+1)
    echo_signals = F.pad(d0_series[:,1:] - d0_series[:, :-1], (1,0))
    delays_us = 2 * spacing * torch.arange(d0_series.shape[1], device=refLR.device) / c

    return echo_signals, delays_us

def compute_gaussian_pulse(refLR: torch.Tensor, spacing: float = 1.0, c: float = 1.54e3, length:int=10, sigma:int=1, pulse=None) -> torch.Tensor:
    """
    Compute the Gaussian pulse for each ray in the batch.

    Parameters
    ----------
    refLR : (B, N)  reflection coeffs for incidence from the left
    spacing : float, spacing between samples in mm
    c : float, speed of sound in m/s

    Returns
    -------
    pulse : (B, N)  Gaussian pulse for each ray
    """
    echo_signals, delays_us = compute_echo_traces(refLR, spacing, c)
    if pulse is None:
        pulse = gaussian_pulse(length=length, sigma=sigma)  
        pulse = torch.tensor(pulse, dtype=echo_signals.dtype, device=echo_signals.device).unsqueeze(0).unsqueeze(0)  # (1, 1, length) to batch on rays
    echo_signals = F.conv1d(echo_signals.unsqueeze(1), pulse, padding=length // 2)  # (B, 1, N)
    
    return echo_signals.squeeze(1)  # (B, N)

def rasterize_fan(x_coords, z_coords, intensities, output_shape=(256, 256)):
    """
    Converts scatter plot data into a 2D image array via interpolation.
    
    Args:
        x_coords, z_coords: 1D arrays of coordinates
        intensities: 1D array of intensities at those points
        output_shape: desired shape of the image (H, W)
        
    Returns:
        2D numpy array of shape (H, W)
    """
    x_coords = np.asarray(x_coords)
    z_coords = np.asarray(z_coords)
    intensities = np.asarray(intensities)

    # Create grid
    grid_x, grid_z = np.meshgrid(x_coords, z_coords)

    # Interpolate intensities onto the grid
    img = griddata(
        points=np.stack((x_coords, z_coords), axis=-1),
        values=intensities,
        xi=(grid_x, grid_z),
        method='linear',
        fill_value=0  # or np.nan
    )
    return img

def gaussian_pulse(length: int, sigma: float) -> np.ndarray:
    """
    Generate a 1D Gaussian pulse centered at 0.
    Parameters
    ----------
    length (int): Total number of points in the pulse (odd for symmetry).
    sigma (float): Standard deviation of the Gaussian (controls width).

    Returns
    -------
    pulse : np.ndarray
        1D array of shape (length,)
    """
    t = np.linspace(-length // 2, length // 2, length)
    pulse = np.exp(-0.5 * (t / sigma) ** 2)
    return pulse / pulse.max()  # normalize to 1


## ARTIFACTS

def radial_falloff_np(image: np.ndarray,
                      attenuation_min: float = 0.999,
                      power: float = 2.0) -> np.ndarray:
    """
    Applique un dégradé d'intensité selon la profondeur (axe samples).
    """
    n_rays, n_samples = image.shape
    # échelle de 1 à attenuation_min, puis puissance
    scale = np.linspace(1.0, attenuation_min, n_samples) ** power
    return image * scale[None, :]

def add_speckle_noise_np(image: np.ndarray,
                         std: float = 0.3) -> np.ndarray:
    """
    Ajoute du speckle noise multiplicatif (bruit gaussien autour de 1).
    """
    noise = np.random.normal(loc=1.0, scale=std, size=image.shape)
    noisy = image * noise
    vmin, vmax = image.min(), image.max()
    return np.clip(noisy, vmin, vmax)

def add_shadow_np(image: np.ndarray,
                  center_ray: int,
                  width: int = 5,
                  strength: float = 0.3) -> np.ndarray:
    """
    Simule une ombre acoustique en atténuant un faisceau (quelques rays).
    """
    shadowed = image.copy()
    start = max(center_ray - width, 0)
    end = min(center_ray + width + 1, image.shape[0])
    shadowed[start:end, :] *= strength
    return shadowed

def sharpen_np(image: np.ndarray,
               alpha: float = 1.5) -> np.ndarray:
    """
    Renforce le contraste local (unsharp masking).
    """
    blurred = gaussian_filter(image, sigma=1)
    sharp = image + alpha * (image - blurred)
    vmin, vmax = image.min(), image.max()
    return np.clip(sharp, vmin, vmax)

def add_speckle_arcs_np(image: np.ndarray,
                              std_radial: float = 0.1,
                              std_local: float = 0.02,
                              power_radial: float = 2.0,
                              power_local: float = 1.5) -> np.ndarray:
    """
    Speckle en arcs dont l'intensité de distorsion augmente avec la profondeur.
    
    - std_radial   : écart‑type de base du bruit radial (créant les arcs).
    - std_local    : écart‑type de base du grain local (texture fine).
    - power_radial : exponent pour renforcer le bruit radial selon la profondeur.
    - power_local  : exponent pour renforcer le grain local selon la profondeur.
    """
    n_rays, n_samples = image.shape
    # 1) Normalisation de la profondeur de 0 (près) à 1 (loin)
    depth_norm = np.linspace(0.0, 1.0, n_samples)

    # 2) STD radial et local variables selon depth_norm
    std_radial_z = std_radial * (1.0 + depth_norm**power_radial)   # arcs de plus en plus marqués
    std_local_z  = std_local  * (1.0 + depth_norm**power_local )   # grain de plus en plus grossier

    # 3) Bruit radial : un facteur par profondeur
    radial_noise = np.random.normal(loc=1.0,
                                    scale=std_radial_z,
                                    size=n_samples)

    # 4) Bruit local : un facteur par pixel (ray × profondeur)
    local_noise = np.random.normal(loc=1.0,
                                   scale=std_local_z[None, :],
                                   size=(n_rays, n_samples))

    # 5) Combinaison multiplicative
    noise = radial_noise[None, :] * local_noise

    # 6) Application et clipping des négatifs
    noised = image * noise
    noised[noised < 0] = 0.0

    return noised

def add_depth_dependent_lateral_blur_np(
    image: np.ndarray,
    max_sigma: float = 2.0
) -> np.ndarray:
    """
    Applique un flou gaussien le long de l'axe latéral (rays) 
    dont l'écart‑type augmente linéairement avec la profondeur.
    """
    n_rays, n_samples = image.shape
    blurred = image.copy()

    for z in range(n_samples):
        sigma = max_sigma * (z / (n_samples - 1)) if z > 0 else 1e-8
        # flou sur chaque colonne (constante en z)
        blurred[:, z] = gaussian_filter1d(blurred[:, z], sigma)

    return blurred

def add_depth_dependent_axial_blur_np(
    image: np.ndarray,
    max_kernel: int = 7
) -> np.ndarray:
    """
    Flou « axial » (sur la profondeur) qui grandit avec z, 
    en moyennant chaque point sur une fenêtre croissante.
    """
    n_rays, n_samples = image.shape
    blurred = image.copy()

    for z in range(n_samples):
        half = int((max_kernel * (z / (n_samples - 1))) // 2)
        if half < 1:
            continue

        start = max(0, z - half)
        end   = min(n_samples, z + half + 1)
        # moyenne la profondeur autour de z
        blurred[:, z] = np.mean(image[:, start:end], axis=1)

    return blurred

def rotate_around_apex(x, 
                       z, 
                       apex, 
                       median):
    """
    Rotate points (x, z) around the apex point to align the median direction with the [0, 1] vector.
    x, z: 1D arrays of coordinates
    apex: (x0, y0) coordinates (float)
    median: (dx, dy) vector representing the median direction (float)
    """
    # Shift points to origin at apex
    device = x.device

    x_shifted = x - 128
    z_shifted = z

    # Compute angle between [0, 1] and median
    median_vec = torch.tensor(median, dtype=torch.float32, device=device)
    median_vec = median_vec / median_vec.norm()
    
    angle = torch.atan2(median_vec[0], median_vec[1])  # angle from [0,1] to median


    # Rotation matrix
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    R = torch.tensor([[cos_a, -sin_a],
                      [sin_a,  cos_a]], device=device)


    # Apply rotation
    coords = torch.stack((x_shifted, z_shifted), dim=0)  # shape (2, N)
    rotated = R @ coords                            # shape (2, N)

    # Shift back to apex-centered
    x_rot = rotated[0] + apex[0]
    z_rot = rotated[1] + apex[1]
    return x_rot, z_rot

def differentiable_splat(x, z, intensities, H=256, W=256, sigma=2.0):
    """
    Efficient differentiable splatting with convolution-based interpolation.
    - x, z must be in [0, W-1] and [0, H-1]
    """
    device = x.device
    x = x.to(dtype=torch.float32)
    z = z.to(dtype=torch.float32)
    intensities = intensities.to(dtype=torch.float32)

    image = torch.zeros((1, 1, H, W), device=device)
    weight = torch.zeros_like(image)

    # Discrete pixel locations
    x_idx = torch.clamp(x.round().long(), 0, W - 1)
    z_idx = torch.clamp(z.round().long(), 0, H - 1)

    # Splat intensity and weight
    image[0, 0, z_idx, x_idx] += intensities
    weight[0, 0, z_idx, x_idx] += 1

    # Define a Gaussian kernel
    size = int(6 * sigma) | 1  # ensure odd size
    coords = torch.arange(size, device=device) - size // 2
    kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] @ kernel_1d[None, :]
    kernel_2d = kernel_2d.to(device).unsqueeze(0).unsqueeze(0)  # shape (1,1,K,K)

    # Blur the image and normalize
    blurred_img = F.conv2d(image, kernel_2d, padding=size//2)
    blurred_weight = F.conv2d(weight, kernel_2d, padding=size//2)
    output = blurred_img / (blurred_weight + 1e-8)

    return output[0, 0]  # Return (H, W) image