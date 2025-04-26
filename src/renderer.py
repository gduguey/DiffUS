import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as ss
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
        return ((Z2 - Z1) / (Z2 + Z1)).pow(2)
        # return np.abs(Z2 - Z1) / (Z1+Z2)

    

    def simulate_rays(self,
        volume: torch.Tensor, 
        source: torch.Tensor, 
        directions: torch.Tensor, 
        num_samples: int = 0) -> torch.Tensor:
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
            self.simulate_ray(volume, source, d)
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
            volume: (D, H, W) Tensor of acoustic properties (e.g., normalized impedance)
            source: (3,) Tensor for the starting point of rays
            directions: (N_rays, 3) Tensor of ray directions (must be unit vectors)
            num_samples: int, number of steps along each ray
            
        Returns:
            R: (N_rays, num_samples-1) Tensor of reflection coefficients along each ray
        """
        # housekeeping
        if directions.ndim == 1:
            directions = directions.unsqueeze(0)
        
        Z = (volume - volume.min()) / (volume.max() - volume.min())
        Z = Z * (1.7e6 - 1.4e6) + 1.4e6

        D, H, W = volume.shape
        
        # Prepare steps and directions
        steps = torch.arange(0, num_samples, dtype=torch.float32, device=volume.device).view(1, -1, 1)  # (1, num_samples, 1)
        directions = directions.unsqueeze(1)  # (N_rays, 1, 3)

        # Trace points
        points = source + steps * directions 

        grid = torch.empty_like(points, dtype=torch.float32)
        
        grid[..., 0] = 2 * (points[..., 2] / max(D - 1, 1)) - 1  # z → x
        grid[..., 1] = 2 * (points[..., 1] / max(H - 1, 1)) - 1  # y → y
        grid[..., 2] = 2 * (points[..., 0] / max(W - 1, 1)) - 1  # x → z

        # Reshape for grid_sample
        grid = grid.view(-1, num_samples, 1, 1, 3) 

        sampler = Z.unsqueeze(0).unsqueeze(0)
        sampler = sampler.expand(grid.shape[0], -1, -1, -1, -1)

        ray_values = F.grid_sample(
            sampler,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
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
            profile = self.trace_ray(volume, src, direction, num_samples)
            all_profiles.append(profile)
        return torch.stack(all_profiles)  # (N, num_samples)

    def plot_beam_frame(
        self,
        volume: torch.Tensor,
        source: torch.Tensor,
        directions: torch.Tensor,
        angle: float = 45.0,
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
            directions=directions
        )

        # 2. Convert to numpy
        frame_np = R.detach().cpu().numpy()  # shape (N_rays, num_samples-1)
        n_rays, n_samples = frame_np.shape

        # 3. Compute ray geometry
        source_2d = np.array([128, 0])  # (x, z), assuming 2D fan centered at (128, 0)
        thetas = np.radians(np.linspace(-angle, angle, n_rays))  # shape (n_rays,)
        ray_len = n_samples  # number of points along each ray

        # Vectorized generation of all points
        steps = np.arange(ray_len)  # (n_samples,)

        directions_xz = np.stack([
            np.sin(thetas),   # (n_rays,)
            np.cos(thetas)    # (n_rays,)
        ], axis=1)  # (n_rays, 2)

        # Expand dimensions
        steps = steps[None, :, None]              # (1, n_samples, 1)
        directions_xz = directions_xz[:, None, :]  # (n_rays, 1, 2)

        points = source_2d[None, None, :] + steps * directions_xz  # (n_rays, n_samples, 2)

        x_coords = points[..., 0].flatten()
        z_coords = points[..., 1].flatten()
        intensities = frame_np.flatten()

        # 4. Plot
        plt.figure(figsize=(6, 6))
        plt.rcParams['axes.facecolor'] = 'black'
        plt.scatter(x_coords, z_coords, c=intensities, cmap='gray', s=1)
        plt.gca().set_aspect('equal')
        plt.xlabel("X")
        plt.ylabel("Z")
        plt.title("Fan-shaped Ultrasound Frame")
        plt.colorbar(label="Intensity")
        plt.show()

    @staticmethod
    def plot_frame(frame: torch.Tensor):
        """
        Display the simulated US frame.
        - frame: (N_rays, depth) intensity map
        """
        plt.figure(figsize=(8, 6))
        # transpose so depth goes downwards
        frame_np = frame.T.cpu().numpy()
        plt.imshow(frame_np, cmap='gray', aspect='auto', vmin=frame_np.min(), vmax=frame_np.max())
        plt.xlabel('Ray index')
        plt.ylabel('Depth sample')
        plt.title('Simulated Ultrasound')
        plt.colorbar(label='Echo intensity')
        plt.show()

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

    def frame_to_bmode(self, raw_frame):
        rf_np = raw_frame.cpu().numpy()
        envelope = np.abs(ss.hilbert(rf_np, axis=1))  # analytic signal
        # envelope = np.abs(2*np.diff(rf_np, axis=1))
        bmode = np.log1p(envelope)  # log compression
        bmode = bmode / np.max(bmode)  # normalize to [0, 1]
        return bmode
        # raw = raw_frame.cpu().numpy()
        # env = np.abs(ss.hilbert(raw, axis=1))
        # bmode = 20*np.log10(env/env.max()+1e-6)
        # bmode = np.clip(bmode, -60, 0)
        # bmode = (bmode + 60)/60
        # return bmode

    def plot_bmode(self, bmode):
        plt.figure(figsize=(8,6))
        bmode_t = bmode.T
        plt.imshow(bmode_t, cmap='gray', aspect='auto', vmin=bmode_t.min(), vmax=bmode_t.max())
        plt.xlabel('Ray #')
        plt.ylabel('Depth sample')
        plt.title('Simulated B-mode Ultrasound')
        plt.colorbar(label='Normalized intensity')
        plt.show()

    def plot_sector_bmode(self,
                        bmode: np.ndarray,
                        angles: np.ndarray,
                        spacing: float = 1.0):
        """
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
        ax.invert_yaxis()
        plt.xlabel('x (lateral)')
        plt.ylabel('z (depth)')
        plt.title('Sector-shaped B-mode Ultrasound Image')
        plt.colorbar(label='Normalized intensity')
        plt.show()

    



