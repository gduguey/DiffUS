"""
Impedance estimation model for ultrasound images. This file contains some source code for the model architecture and training procedures. The model is a Basic, 32 hidden layer MLP that takes normalized intensity values as input and outputs estimated acoustic impedance.

"""
import torch
import torch.nn.functional as F
import warnings
from tqdm import tqdm

import torch.nn as nn
import torch
from .utils import create_brain_mask, zscore_normalize

import torch.optim as optim

class ImpedanceEstimator(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.min_MRI = 0.0
        self.max_MRI = 1.0
    def forward(self, x):
        # Normalize input
        x = self.max_MRI * (x - torch.min(x)) / (torch.max(x) - torch.min(x)) + self.min_MRI
        return self.net(x)
    
    @staticmethod
    def train_model(X, y, input_dim=1, epochs=1000, lr=0.01):
        # add a scaler
        model = ImpedanceEstimator(input_dim)
        model.min_MRI = X.min().item()
        model.max_MRI = X.max().item()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        all_loss = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            all_loss.append(loss.item())
            if (epoch+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4e}')
                
        return model, all_loss

    @staticmethod
    def compute_impedance_volume(
        volume: torch.Tensor,
        model: 'ImpedanceEstimator',
        threshold: float = 50
    ) -> torch.Tensor:
        """Generate full impedance volume using trained model and preprocessing."""
        mask = create_brain_mask(volume, threshold)
        vol_norm = zscore_normalize(volume.float(), mask)
        
        with torch.no_grad():
            Z_pred = model(vol_norm[mask].unsqueeze(1)).squeeze() * 1e6
            
        Z_vol = torch.full_like(volume, 400.0)  # Default air impedance
        Z_vol[mask] = Z_pred
        return Z_vol