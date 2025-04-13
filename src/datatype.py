import jax.numpy as jnp
import sys
import pathlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import nibabel as nib
import torch
from torch.utils.data import Dataset
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
            plt.imshow(img.numpy(), cmap='coolwarm')
            plt.title(f"Slice {slice_id} of {self.name}, orientation {orientation}")
            plt.colorbar()
            plt.axis("off")
            plt.show()

class MRIDataset(MedicalVolumeDataset):
    def __init__(self, path, name="MRI", axis=0, dtype=torch.float32):
        super().__init__(path, name, axis, dtype)

    def plot3D(self, threshold=0.2):
        volume = (self.data - self.data.min()) / (self.data.max() - self.data.min())
        volume_np = volume.numpy()
        
        fig = go.Figure(data=go.Isosurface(
            x=np.arange(volume_np.shape[0]).repeat(volume_np.shape[1]*volume_np.shape[2]),
            y=np.tile(np.arange(volume_np.shape[1]).repeat(volume_np.shape[2]), volume_np.shape[0]),
            z=np.tile(np.arange(volume_np.shape[2]), volume_np.shape[0]*volume_np.shape[1]),
            value=volume_np.flatten(),
            isomin=0.4,
            isomax=1.0,
            opacity=0.1,
            surface_count=2,
            caps=dict(x_show=False, y_show=False)
        ))

        fig.update_layout(scene=dict(aspectmode='data'))
        fig.show()

class iUSDataset(MedicalVolumeDataset):
    def __init__(self, path, name="iUS", axis=0, dtype=torch.float32):
        super().__init__(path, name, axis, dtype)
