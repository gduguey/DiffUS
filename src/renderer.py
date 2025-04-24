import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as ss

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
        # return ((Z2 - Z1) / (Z2 + Z1)).pow(2)
        return np.abs(Z2 - Z1)

    def simulate_ray(self,
                     volume: torch.Tensor,
                     source: torch.Tensor,
                     direction: torch.Tensor) -> torch.Tensor:
        """
        Simulate one A line:
        - volume: (D,H,W) tensor of impedances
        - source: (3,) starting point in voxel coords (x,y,z)
        - direction: (3,) normalized direction vector
        Returns a (num_samples - 1,) tensor of echo intensities.
        """
        # sample impedances along the ray
        impedances = self.trace_ray(volume, source, direction, self.num_samples)  # (num_samples,)
        # compute R at each interface
        Z1, Z2 = impedances[:-1], impedances[1:]
        R = self.compute_reflection_coeff(Z1, Z2)  # (num_samples - 1,)

        # attenuation proportional to exp(-alpha * depth). depth index starts at 1
        # depths = torch.arange(1, self.num_samples, dtype=impedances.dtype)
        # attenuation = torch.exp(-self.attenuation_coeff * depths)

        # echo intensity profile
        # return R * attenuation  # (num_samples - 1,)
        return R

    def simulate_frame(self,
                       volume: torch.Tensor,
                       source: torch.Tensor,
                       directions: torch.Tensor) -> torch.Tensor:
        """
        Simulate a full ultrasound frame.
        - directions: (N_rays, 3) each row a unit vector
        Returns: (N_rays, num_samples - 1) intensity map
        """
        return torch.stack([
            self.simulate_ray(volume, source, d)
            for d in directions
        ], dim=0)
    
    @staticmethod
    def trace_ray(
        volume: torch.Tensor,
        source: torch.Tensor,
        direction: torch.Tensor,
        num_samples: int
    ) -> torch.Tensor:
        """
        Trace a ray through a 3D volume and sample intensity values.

        Args:
            volume (torch.Tensor): 3D volume of shape (D, H, W).
            source (torch.Tensor): Starting point of the ray (x, y, z).
            direction (torch.Tensor): Direction of the ray (dx, dy, dz).
            num_samples (int): Number of samples to take along the ray.

        Returns:
            torch.Tensor: Sampled intensity values along the ray.
        """
        # Get volume dimensions      

        volume = (volume - volume.min()) / (volume.max() - volume.min())
        D, H, W = volume.shape
        Z = volume * (1.7e6 - 1.4e6) + 1.4e6

        # Normalize direction
        direction = direction / direction.norm()

        # Compute points along the ray
        steps = torch.arange(0, num_samples, dtype=torch.float32).unsqueeze(1)
        points = source + steps * direction  # (num_samples, 3)

        # Normalize to [-1, 1] for grid_sample
        grid = torch.empty_like(points)
        grid[:, 0] = 2 * (points[:, 2] / max(D - 1, 1)) - 1  # z → x
        grid[:, 1] = 2 * (points[:, 1] / max(H - 1, 1)) - 1  # y → y
        grid[:, 2] = 2 * (points[:, 0] / max(W - 1, 1)) - 1  # x → z
        grid = grid.view(1, num_samples, 1, 1, 3)  # (N, D_out, H_out, W_out, 3)

        # Prepare volume for grid_sample: (N, C, D, H, W)
        Z = Z.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

        # Sample
        ray_values = F.grid_sample(Z, grid, align_corners=True, mode='bilinear').squeeze()

        return ray_values # (num_samples,)
    
    def trace_rays(self, volume, sources, direction, num_samples):
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

    



