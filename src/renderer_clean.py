import torch
import torch.nn.functional as F
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d

class UltrasoundRenderer:
    def __init__(self, num_samples: int, attenuation_coeff: float = 0.5, verbose: bool = False):
        """
        num_samples: how many points to sample along each ray
        attenuation_coeff: controls exponential decay of echoes with depth
        """
        self.num_samples = num_samples
        self.attenuation_coeff = attenuation_coeff
        self.verbose = verbose

    @staticmethod
    def compute_reflection_coeff(Z1: torch.Tensor, Z2: torch.Tensor) -> torch.Tensor:
        """
        Compute power reflection coefficient between two impedances. This is given by the formula: R = (Z2 - Z1) / (Z2 + Z1)
        R = ((Z2 - Z1)/(Z2 + Z1))**2
        """
        return (Z2 - Z1) / (Z1 + Z2)

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
                
        x,y,z, impedances =  self.trace_ray(
            volume=volume, 
            source=source, 
            directions=directions, 
            num_samples=num_samples,
            start=start)
        if impedances.ndim == 1:
            impedances = impedances.unsqueeze(0)
        Z1 = impedances[:, :-1]  # (num_samples-1)
        Z2 = impedances[:, 1:]   # (num_samples-1)

        R = self.compute_reflection_coeff(Z1, Z2)
        if MRI:
            return Z1
        return x,y,z, R.squeeze(0) 
    
    @staticmethod
    def trace_ray(
                volume: torch.Tensor, 
                source: torch.Tensor, 
                directions: torch.Tensor, 
                num_samples: int, start: int,
                verbose: bool = False) -> torch.Tensor:
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
        if verbose:
            print("[INFO] Tracing rays with source:", source, "and directions shape:", directions.shape)
        # Trace points
        points = source + steps * directions 

        x,y,z, ray_values = custom_nearest_sampler(volume, points, start=start)
        if verbose:
            print("[INFO] Ray values shape:", ray_values.shape)
        return x,y,z, ray_values
    
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

    def render_us_frame(
        self,
        volume: torch.Tensor,
        source: torch.Tensor,
        directions: torch.Tensor,
        angle: float = 45.0,
        plot: bool = True,
        artifacts: bool = False,
        std_radial: float = 0.01,
        std_local: float = 0.15,
        max_sigma: float = 4.0,
        alpha: float = 5,
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
        x,y,z, R = self.simulate_rays(
            volume=volume,
            source=source,
            directions=directions,
            MRI=False,
            start=start
        ) # This samples the volume, computes reflection coefficients and gives the geometry of the rays
        device = R.device
        
        # 2. Truncation (if start is offset)
        # housekeeping
        if type(start) is float:
            start = int(start * self.num_samples)
        if type(start) is int:
            start = max(0, start)  # ensure non-negative index

        
        
        if start > 0:
            R = R[:, start:]
            median_first = R[:, 0].median()
            R[:, 0] = median_first
            if self.verbose:
                print("[INFO] Starting from sample index:", start, "(for instance, to skip bones)")

        if self.verbose:
            print("[INFO] Sampling complete. Samples shape:", R.shape)
        # 3. Process the output to propagate the rays
        processed_output, _ = compute_echo_traces(R, verbose = self.verbose)
        if self.verbose:
            print("[INFO] Processed output shape:", processed_output.shape)

        # 4. Attenuation model
        processed_output = attenuate(processed_output, 
                                     attenuation_coeff=self.attenuation_coeff,
                                     device=device)

        # 5. Apply artifacts if requested
        if artifacts:
            # speckle
            processed_output = add_speckle_arcs_np(processed_output, 
                                            std_radial=std_radial,
                                            std_local=std_local)
            # lateral blur
            processed_output = add_depth_dependent_lateral_blur_np(processed_output,
                                            max_sigma=max_sigma)
            # sharpen
            processed_output = sharpen_np(processed_output, 
                                            alpha=alpha)   
        
        return x[:,start:], y[:,start:], z[:,start:], processed_output
    
    @staticmethod
    def plot_frame(frame: torch.Tensor, ax: plt.Axes = None):
        """
        Display the given frame.
        - frame: (N_rays, depth) intensity map
        - ax: optional matplotlib Axes to plot on

        If ax is None, creates a new figure.
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
                    ray_map_frame: torch.Tensor,
                    angles: torch.Tensor,
                    spacing: float = 1.0):
        """
        Plot a ray map of the US frame in true sector geometry.
        - frame: (N_rays, depth) intensity map
        - angles: (N_rays,) ray angles in radians
        - spacing: real-world depth per sample (e.g. mm per voxel)
        """
        N_rays, depth = ray_map_frame.shape
        depths = torch.arange(0, depth, dtype=ray_map_frame.dtype) * spacing

        # gather (x,z) coords + intensity
        xs, zs, vals = [], [], []
        for i, theta in enumerate(angles):
            sin, cos = torch.sin(theta), torch.cos(theta)
            for j, r in enumerate(depths):
                xs.append((r * sin).item())
                zs.append((r * cos).item())
                vals.append(ray_map_frame[i, j].item())

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
    
# UTILITIES

def prop_single_ray(refLR: torch.Tensor, 
                    traLR: torch.Tensor = None, 
                    traRL: torch.Tensor = None, 
                    verbose:bool=False) -> torch.Tensor:
    """
    Solve a 2(N+1)*2(N+1) system for *each* ray in the batch. Given the refLR reflection coefficients list, this function computes a matrix of size (B,2(N+1),2(N+1)). and returns the solution vector (shape (B, 2*(N+1))).

    Parameters
    ----------
    refLR : (B, N)  reflection coeffs for incidence from the left
    traLR : (B, N)  transmission coeffs for incidence from the left
    traRL : (B, N)  transmission coeffs for incidence from the right

    Returns
    -------
    w : (B, 2*(N+1)) laid out as [g0, d0, g1, d1, …, gN, dN]
    """
    B, N  = refLR.shape
    if not traLR: traLR = 1 + refLR              # t_il
    if not traRL: traRL = 1 - refLR              # t_ir
    refRL = refLR                 # r_rl  (OK only if impedances equal?)

    size = 2 * (N + 1)
    A = torch.zeros((B, size, size), dtype=refLR.dtype, device=refLR.device)
    b = torch.zeros((B, size),      dtype=refLR.dtype, device=refLR.device)

    # boundary conditions: g0 = 1 , d_{N+1}=0 (input energy at the source)
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
    w = torch.nan_to_num(w, nan=0.0)
    if verbose:
        print("[INFO] Propagated single ray solution shape:", w.shape)
        print("\tA matrix shape:", A.shape)
        print("\tb vector shape:", b.shape)
        print("\tw vector shape:", w.shape)

    return w

def propagate_full_rays_batched(refLR: torch.Tensor, verbose: bool = False) -> torch.Tensor:
    """
    For each truncation depth i = 0...N, compute the surface-return amplitude d0^{(i)}. This is an iterative process that solves the system of equations for each depth, starting from the first reflection coefficient and propagating through the series of reflections.

    Parameters
    ----------
    refLR : (B, N)  reflection coeffs for incidence from the left

    Returns
    -------
    d0_per_depth : (B, N+1)  with d0^{(0)}, d0^{(1)}, …, d0^{(N)}
    """
    if verbose:
        print("[INFO] Propagating full rays for", refLR.shape[0], "rays with", refLR.shape[1], "depths.")
    B, N = refLR.shape

    d0_per_depth = []
    total = []
    for i in range(N + 1):
        w = prop_single_ray(refLR[:, :i])        # solve the truncated system
        d0_per_depth.append(w[:, 1])       # column 1 is d0
        total.append(w)
    # return total
    if verbose:
        print("[INFO] Stacked total shape:", '\n'.join([str(t.shape) for t in total]))

    stacked_d0 = torch.stack(d0_per_depth, dim=1)
    stacked_d0 = torch.cumsum(stacked_d0, dim=1)  # cumulative sum along the depth axis
    return stacked_d0



def compute_echo_traces(refLR: torch.Tensor, spacing: float = 1.0, c: float = 1.54e3, verbose=False) -> torch.Tensor:
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

    d0_series = propagate_full_rays_batched(refLR, verbose=verbose)  # (B, N+1)
    if verbose:
        print("[INFO] Computed d0 series shape:", d0_series.shape)
    
    echo_signals = F.pad(d0_series[:,1:] - d0_series[:, :-1], (1,0))
    if verbose:
        print("[INFO] Echo signals shape after padding:", echo_signals.shape)
    
    delays_us = 2 * spacing * torch.arange(d0_series.shape[1], device=refLR.device) / c
    if verbose:
        print("[INFO] Delays in microseconds shape:", delays_us.shape)
    return echo_signals, delays_us

def compute_gaussian_pulse(refLR: torch.Tensor, spacing: float = 1.0, c: float = 1.54e3, length:int=10, sigma:int=1, pulse=None, verbose=False) -> torch.Tensor:
    """
    Compute the Gaussian pulse for each ray in the batch. This is an alternate method to soften the reflection coefficients and simulate the ultrasound pulse.

    Parameters
    ----------
    refLR : (B, N)  reflection coeffs for incidence from the left
    spacing : float, spacing between samples in mm
    c : float, speed of sound in m/s

    Returns
    -------
    pulse : (B, N)  Gaussian pulse for each ray
    """

    echo_signals, delays_us = compute_echo_traces(refLR, spacing, c, verbose=verbose)
    
    if pulse is None:
        pulse = gaussian_pulse(length=length, sigma=sigma)  
        pulse = torch.tensor(pulse, dtype=echo_signals.dtype, device=echo_signals.device).unsqueeze(0).unsqueeze(0)  # (1, 1, length) to batch on rays
    echo_signals = F.conv1d(echo_signals.unsqueeze(1), pulse, padding=length // 2)  # (B, 1, N)
    
    return echo_signals.squeeze(1)  # (B, N)

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

######################
##### UTILITIES ######
######################

def rasterize_fan(x_coords, z_coords, intensities, output_shape=(256, 256)):
    """
    Converts scatter plot data into a 2D image array via interpolation.
    
    Parameters
    ----------
    x_coords : np.ndarray
        1D array of x-coordinates.
    z_coords : np.ndarray
        1D array of z-coordinates.
    intensities : np.ndarray
        1D array of intensities at the corresponding (x, z) points.
    output_shape : tuple of int, optional
        Desired shape of the output image (H, W). Default is (256, 256).
        
    Returns
    -------
    np.ndarray
        2D array of shape (H, W) representing the interpolated image.
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

def rotate_around_apex(x, 
                       z, 
                       apex, 
                       median):
    """
    Rotate points (x, z) around the apex point to align the median direction with the [0, 1] vector.
    
    Parameters
    ----------
    x : torch.Tensor
        1D tensor of x-coordinates.
    z : torch.Tensor
        1D tensor of z-coordinates.
    apex : tuple of float
        (x0, z0) coordinates of the apex point.
    median : tuple of float
        (dx, dz) vector representing the median direction.

    Returns
    -------
    x_rot : torch.Tensor
        Rotated x-coordinates.
    z_rot : torch.Tensor
        Rotated z-coordinates.
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

def differentiable_splat(x, y, z, intensities, H=256, W=256, sigma=2.0, verbose=False):
    """
    Differentiable splatting onto the 2D plane of highest variance. This helps locate the orientation and recover the correct way to plot the resulting image.
    
    Parameters
    ----------
    x : torch.Tensor
        Tensor of x-coordinates in [0, size-1].
    y : torch.Tensor
        Tensor of y-coordinates in [0, size-1].
    z : torch.Tensor
        Tensor of z-coordinates in [0, size-1].
    intensities : torch.Tensor
        Tensor of intensity values, same shape as x, y, z.
    H : int
        Output image height (for axis0).
    W : int
        Output image width (for axis1).
    sigma : float
        Standard deviation for Gaussian kernel used in splatting.

    Returns
    -------
    torch.Tensor
        2D image of shape (H, W) with splatted intensities.
    """
    device = intensities.device
    coords = [x, y, z]
    # Compute variance for each axis
    variances = [c.float().var().item() for c in coords]
    if verbose:
        print(f"[INFO] Variances: {variances}")
    # Get indices of two axes with highest variance
    axis0, axis1 = sorted(range(3), key=lambda i: -variances[i])[:2]

    coord0 = coords[axis0].to(dtype=torch.float32)
    coord1 = coords[axis1].to(dtype=torch.float32)
    intensities = intensities.to(dtype=torch.float32)

    image = torch.zeros((1, 1, H, W), device=device)
    weight = torch.zeros_like(image)

    # Discrete pixel locations
    idx0 = torch.clamp(coord0.round().long(), 0, W - 1)
    idx1 = torch.clamp(coord1.round().long(), 0, H - 1)

    # Splat intensity and weight
    image[0, 0, idx1, idx0] += intensities
    weight[0, 0, idx1, idx0] += 1

    # Define a Gaussian kernel
    size = int(6 * sigma) | 1  # ensure odd size
    coords_kernel = torch.arange(size, device=device) - size // 2
    kernel_1d = torch.exp(-0.5 * (coords_kernel / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] @ kernel_1d[None, :]
    kernel_2d = kernel_2d.to(device).unsqueeze(0).unsqueeze(0)  # shape (1,1,K,K)

    # Blur the image and normalize
    blurred_img = F.conv2d(image, kernel_2d, padding=size//2)
    blurred_weight = F.conv2d(weight, kernel_2d, padding=size//2)
    output = blurred_img / (blurred_weight + 1e-8)

    return output[0, 0].T

## Sampler

def custom_nearest_sampler(Z, points, visualize=True, sampler='custom', start=100, verbose=False):
    """
    Sample values from a 3D volume Z at specified points using nearest neighbor sampling. This function supports both 'custom' sampling for the hand-coded sampler and 'grid_sample' for torch's legacy sampler.

    Args:
        Z: torch.Tensor of shape (D, H, W)
        points: torch.Tensor of shape (..., 3), in pixel coordinates (x, y, z)
        visualize: bool, whether to show a visualization of the sampled points. The plane to plot one is chosen through variance minimization.
        sampler: str, 'custom' for the hand-coded sampler, 'grid_sample' for torch's grid_sample
        start: int, starting index for truncation (e.g., to skip bones)
        verbose: bool, whether to print debug information
    Returns:
        ray_values: torch.Tensor of shape (...), sampled values
    """
    D, H, W = Z.shape
    points = points.float()
    batch_size, num_samples, _ = points.shape
        
    x = torch.clamp(points[..., 0].round().long().flatten(), 0, D - 1)
    y = torch.clamp(points[..., 1].round().long().flatten(), 0, H - 1)
    z = torch.clamp(points[..., 2].round().long().flatten(), 0, W - 1)
    
    if sampler == 'prop':
        values = Z[x,y,z]  # shape: [batch_size * num_samples]
        ray_values = values.view(batch_size, num_samples)
    
    if sampler != 'prop':
        # Ensure Z is in (D, H, W) order
        Z = Z.unsqueeze(0).unsqueeze(0).contiguous().float()  # (1, 1, D, H, W)
        grid = torch.empty_like(points, dtype=torch.float32, device=Z.device)
        grid[..., 0] = 2 * (points[..., 2] / (W - 1)) - 1  # x
        grid[..., 1] = 2 * (points[..., 1] / (H - 1)) - 1  # y
        grid[..., 2] = 2 * (points[..., 0] / (D - 1)) - 1  # z

        grid = grid.view(1, batch_size, num_samples, 1, 3)
        if verbose:  # DEBUG
            print("[INFO] Grid shape:", grid.shape, "Z shape:", Z.shape)
        ray_values = F.grid_sample(
            Z, grid, mode='nearest'
        ).squeeze()

    if visualize:
        # Compute variance to pick projection plane
        print("[INFO] Visualizing sampled points in 3D volume")
        var_x = x.float().var().item()
        var_y = y.float().var().item()
        var_z = z.float().var().item()
        variances = [var_x, var_y, var_z]
        print(f"[INFO] Variances: x={var_x:.4f}, y={var_y:.4f}, z={var_z:.4f}")
        drop_axis = variances.index(min(variances))
        axis_names = ['x', 'y', 'z']
        keep_axes = [i for i in range(3) if i != drop_axis]
        ax0, ax1 = keep_axes
        x_ex = x.view(batch_size, num_samples)[:, start:].flatten()
        y_ex = y.view(batch_size, num_samples)[:, start:].flatten()
        z_ex = z.view(batch_size, num_samples)[:, start:].flatten()
        mapping = {'x': x_ex, 'y': y_ex, 'z': z_ex}

        proj0 = mapping[axis_names[ax0]].flatten().cpu()
        proj1 = mapping[axis_names[ax1]].flatten().cpu()
        sampled_vals = ray_values[:, start:].flatten().cpu()

        vol_np = Z.cpu().numpy()
        slice_idx = int(points.flatten(0, -2)[0, drop_axis].item())
        slice_idx = max(0, min(slice_idx, Z.shape[drop_axis] - 1))

        if drop_axis == 0:
            slice_img = vol_np[slice_idx, :, :]
        elif drop_axis == 1:
            slice_img = vol_np[:, slice_idx, :]
        else:
            slice_img = vol_np[:, :, slice_idx]

        plt.figure(figsize=(4, 4))
        print(slice_img.shape)
        plt.imshow(slice_img.squeeze(), cmap='gray', alpha=1, origin='lower')
        plt.scatter(proj1, proj0, s=8, c=sampled_vals, cmap='jet', alpha=0.4)
        plt.title(f"Sampled points in {axis_names[ax0]}-{axis_names[ax1]} plane")
        plt.xlabel(axis_names[ax0])
        plt.ylabel(axis_names[ax1])
        plt.axis("off")
        plt.show()
    
    # reshape outputs to (batch_size, num_samples)
    x = x.view(batch_size, num_samples)
    y = y.view(batch_size, num_samples)
    z = z.view(batch_size, num_samples)
    return x,y,z, ray_values

#####################
##### ARTIFACTS #####
#####################

def attenuate(processed_output, attenuation_coeff, device):
    """
    Apply exponential attenuation to the processed output.
    Parameters
    ----------
    processed_output : torch.Tensor
        The output tensor to apply attenuation to, shape (N_rays, num_samples).
    attenuation_coeff : float
        The coefficient for exponential decay.
    device : torch.device
        The device on which the tensor is located.
    
    Returns
    -------
    torch.Tensor
        The processed output with applied attenuation, shape (N_rays, num_samples).
    """
    depths = torch.arange(processed_output.shape[1], device=device).float()
    attenuation = torch.exp(-attenuation_coeff * depths)  # shape (num_samples,)
    processed_output = processed_output * attenuation[None, :]  # shape (N_rays, num_samples)
    return processed_output

def radial_falloff_np(image: np.ndarray,
                      attenuation_min: float = 0.999,
                      power: float = 2.0) -> np.ndarray:
    """
    
    Applies a radial falloff to the image, simulating attenuation with depth.
    Parameters
    ----------
    - image: (N_rays, depth) intensity map
    - attenuation_min: minimum attenuation factor at the farthest depth
    - power: exponent controlling the rate of attenuation
    
    Returns
    -------
    np.ndarray
        The image with applied radial falloff.
    """
    n_rays, n_samples = image.shape
    # échelle de 1 à attenuation_min, puis puissance
    scale = np.linspace(1.0, attenuation_min, n_samples) ** power
    return image * scale[None, :]

def add_speckle_noise_np(image: np.ndarray,
                         std: float = 0.3) -> np.ndarray:
    """
    Add multiplicative speckle noise (Gaussian noise around 1). 
    Parameters
    ----------
    - image: (N_rays, depth) intensity map
    - std: standard deviation of the Gaussian noise (1.0 = no noise, 0.3 = moderate noise)

    Returns
    -------
    np.ndarray
        The image with added speckle noise.
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
    Simulate an acoustic shadow by attenuating a beam (a few rays).
    Parameters
    ----------
    - image: (N_rays, depth) intensity map
    - center_ray: index of the ray where the shadow starts
    - width: number of rays to shadow on each side of the center_ray
    - strength: factor to reduce the intensity in the shadowed region (0.0 = no shadow, 1.0 = full shadow)

    Returns
    -------
    np.ndarray
        The image with applied shadow effect.
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
    Parameters
    ----------
    - image: (N_rays, depth) intensity map
    - alpha: sharpening factor (1.0 = no sharpening, >1.0 = more sharpening)

    Returns
    -------
    np.ndarray
        The sharpened image.
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
    Add speckle arcs with distortion intensity increasing with depth.
    
    Parameters
    ----------
    - std_radial : float
        Base standard deviation for radial noise (creating arcs).
    - std_local : float
        Base standard deviation for local grain (fine texture).
    - power_radial : float
        Exponent to amplify radial noise with depth.
    - power_local : float
        Exponent to amplify local grain with depth.

    Returns
    -------
    np.ndarray
        The image with added speckle arcs.
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
    Adds lateral blur that increases with depth, simulating the effect of ultrasound beam divergence.

    Parameters
    ----------
    - image: (N_rays, depth) intensity map
    - max_sigma: maximum standard deviation for the Gaussian blur at the deepest sample
    Returns
    -------
    np.ndarray
        The image with added lateral blur.
    """
    n_rays, n_samples = image.shape
    blurred = image.numpy()

    for z in range(n_samples):
        sigma = max_sigma * (z / (n_samples - 1)) if z > 0 else 1e-8
        # flou sur chaque colonne (constante en z)
        blurred[:, z] = gaussian_filter1d(blurred[:, z], sigma)

    return torch.from_numpy(blurred)

def add_depth_dependent_axial_blur_np(
    image: np.ndarray,
    max_kernel: int = 7
) -> np.ndarray:
    """
    Adds axial blur that increases with depth, simulating the effect of ultrasound beam divergence.
    Parameters
    ----------
    - image: (N_rays, depth) intensity map
    - max_kernel: maximum kernel size for the blur at the deepest sample
    Returns
    -------
    np.ndarray
        The image with added axial blur.
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