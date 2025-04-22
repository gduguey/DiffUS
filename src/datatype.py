import jax.numpy as jnp
import sys
import pathlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import nibabel as nib
import torch
from torch.utils.data import Dataset
import torchio as tio
import numpy as np

pio.renderers.default = 'browser'

## SETUP
PROJECT_ROOT = pathlib.Path().cwd().parent
sys.path.append(str(PROJECT_ROOT))

## DATATYPE CLASS
# This class is used to load and manipulate MRI data

class MedicalVolumeDataset(Dataset):
    def __init__(self, path, name, axis=0, dtype=torch.float32):
        self.path = path
        self.name = name
        self.axis = axis
        self.dtype = dtype

        try:
            self.data = torch.tensor(nib.load(path).get_fdata(), dtype=self.dtype)
        except Exception as e:
            raise ValueError(f"Could not load data from {path} with name {name}. Error: {e}")

        self.num_slices = self.data.shape[self.axis]

    def __len__(self):
        return self.num_slices

    def __getitem__(self, idx):
        if self.axis == 0:
            slice_ = self.data[:, :, idx]
        elif self.axis == 1:
            slice_ = self.data[:, idx, :]
        elif self.axis == 2:
            slice_ = self.data[idx, :, :]
        else:
            raise ValueError(f"Invalid axis {self.axis}. Must be 0, 1, or 2.")

        slice_ = (slice_ - slice_.min()) / (slice_.max() - slice_.min() + 1e-5)
        return slice_.unsqueeze(0)  # (1, H, W)

    def plot2D(self, slice_id=0, axis=None, plot=True):
        axis = self.axis if axis is None else axis
        if axis == 0:
            img = self.data[:, :, slice_id]
        elif axis == 1:
            img = self.data[:, slice_id, :]
        elif axis == 2:
            img = self.data[slice_id, :, :]
        else:
            raise ValueError("Axis must be 0, 1, or 2.")

        orientation = {0: 'axial', 1: 'coronal', 2: 'sagittal'}.get(axis, 'unknown')
        if plot:
            plt.imshow(img.numpy(), cmap='gray')
            plt.title(f"Slice {slice_id} of {self.name}, orientation {orientation}")
            plt.colorbar()
            plt.axis("off")
            plt.show()

class MRIDataset(Dataset):
    def __init__(self, paths, name="MRI", axis=0, dtype=torch.float32):
        self.name = name
        self.axis = axis
        self.dtype = dtype
        self.subjects = [
            tio.Subject({name: tio.ScalarImage(path)}) for path in paths
        ]
        self.dataset = tio.SubjectsDataset(self.subjects)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        subject = self.dataset[idx]
        image_tensor = subject[self.name].data
        affine = subject[self.name].affine
        spacing = subject[self.name].spacing
        return {
            'image': image_tensor,
            'affine': affine,
            'spacing': spacing,
            'path': subject[self.name].path
        }
    
    def plot3D(self, threshold=0.2, idx=0):
        sample = self[idx]
        if sample is None:
            print("Cannot plot: sample is None.")
            return

        volume = sample['image'].squeeze(0)  # remove channel dim: (D, H, W)
        volume = (volume - volume.min()) / (volume.max() - volume.min())
        volume_np = volume.cpu().numpy()

        D, H, W = volume_np.shape
        x, y, z = np.meshgrid(np.arange(W), np.arange(H), np.arange(D), indexing='ij')

        fig = go.Figure(data=go.Isosurface(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=volume_np.flatten(),
            isomin=threshold,
            isomax=1.0,
            opacity=0.1,
            surface_count=2,
            caps=dict(x_show=False, y_show=False, z_show=False)
        ))

        fig.update_layout(scene=dict(aspectmode='data'))
        fig.show()
    
    def plot2D(self, idx=0, slice_id=0, axis=None, plot=True):
        sample = self[idx]
        if sample is None:
            print("Cannot plot: sample is None.")
            return

        axis = self.axis if axis is None else axis
        volume = sample['image'].squeeze(0)  # shape: (D, H, W)

        if axis == 0:
            img = volume[slice_id, :, :]
        elif axis == 1:
            img = volume[:, slice_id, :]
        elif axis == 2:
            img = volume[:, :, slice_id]
        else:
            raise ValueError("Axis must be 0 (axial), 1 (coronal), or 2 (sagittal).")

        orientation = {0: 'axial', 1: 'coronal', 2: 'sagittal'}.get(axis, 'unknown')
        
        if plot:
            plt.imshow(img.numpy(), cmap='gray')
            plt.title(f"{self.name} - Slice {slice_id} ({orientation})")
            plt.axis("off")
            plt.colorbar()
            plt.show()

        return img

    def plot_voxels(self, idx=0, threshold=0.5):
        sample = self[idx]
        volume = sample['image'].squeeze(0)  # (D, H, W)
        volume = (volume - volume.min()) / (volume.max() - volume.min())
        binary_volume = (volume > threshold).cpu().numpy()

        D, H, W = binary_volume.shape
        x, y, z = np.nonzero(binary_volume)

        fig = go.Figure(data=go.Scatter3d(
            x=z, y=y, z=x,  # note: numpy axis ordering
            mode='markers',
            marker=dict(
                size=2,
                color='blue',
                opacity=0.2,
            )
        ))
        fig.update_layout(scene=dict(aspectmode='data'))
        fig.show()

class iUSDataset(MedicalVolumeDataset):
    def __init__(self, path, name="iUS", axis=0, dtype=torch.float32):
        super().__init__(path, name, axis, dtype)
