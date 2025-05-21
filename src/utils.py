import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion


### TREATMENT FUNCTIONS

def create_brain_mask(volume, threshold=50):
    """
    Quick brain mask generation by thresholding and cleaning.
    volume: np.ndarray, MRI volume
    threshold: int, intensity below which is considered air/background
    """
    mask = volume > threshold
    mask = binary_dilation(mask, iterations=2) # pourquoi?
    mask = binary_erosion(mask, iterations=2) # pourquoi? 
    return torch.from_numpy(mask)

def zscore_normalize(volume, mask):
    """
    Normalize a volume by z-scoring inside a brain mask.
    Args:
        volume: torch.Tensor, the 3D volume to normalize.
        mask: torch.Tensor, binary brain mask of the same shape as the volume.
    
    Returns:
        torch.Tensor: The z-score normalized volume.
    """
    volume = volume.float()
    brain_voxels = volume[mask > 0]
    mean = brain_voxels.mean()
    std = brain_voxels.std()
    
    volume_norm = (volume - mean) / (std + 1e-8)  
    return volume_norm

## PLOTTING FUNCTIONS

def plot_histogram(volume):
    plt.figure(figsize=(12, 6))

    plt.hist(volume.flatten(), bins=50, color='blue', alpha=0.7)
    plt.title('T1 Volume Intensity Distribution')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    # plt.xlim(0,20)

    plt.tight_layout()
    plt.show()

def render_video(triplet_list, xlim=(0, 1), ylim=(0, 1), cmap='viridis', interval=100):
    """
    Animate a list of (x, y, intensity) frames as a scatter video.
    """
    fig, ax = plt.subplots()
    x0, y0, i0 = triplet_list[0]

    sc = ax.scatter(x0, y0, c=i0, s=1, cmap=cmap, vmin=min(i0), vmax=max(i0))
    ax.set_facecolor('black')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    title = ax.set_title("Frame 0")
    

    def animate(i):
        x, y, intensity = triplet_list[i]
        sc.set_offsets(np.column_stack((x, y)))
        ax.set_facecolor('black')
        sc.set_array(intensity)
        ax.set_xticks([])
        ax.set_yticks([])
        sc.set_clim(vmin=min(intensity), vmax=max(intensity))
        title.set_text(f"Frame {i}")
        return sc,

    plt.close(fig)  # Prevents extra figure in notebooks
    ani = animation.FuncAnimation(
        fig, animate, frames=len(triplet_list), interval=interval, blit=False
    )
    return ani

def render_video_frame(frames, xlim=(0, 1), ylim=(0, 1), cmap='grey', interval=100):
    """
    Animate a list of (2D Image) frames as a scatter video.
    """
    fig, ax = plt.subplots()
    
    sc = ax.imshow(frames[0], cmap=cmap, vmin=torch.min(frames[0]), vmax=torch.max(frames[0]))

    
    # ax.set_xlim(*xlim)
    # ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    title = ax.set_title("Frame 0")

    def animate(i):
        x = frames[i]
        sc.set_array(x)
        ax.set_xticks([])
        ax.set_yticks([])
        sc.set_clim(vmin=torch.min(x), vmax=torch.max(x))
        title.set_text(f"Frame {i}")
        return sc,

    plt.close(fig)  # Prevents extra figure in notebooks
    ani = animation.FuncAnimation(
        fig, animate, frames=len(frames), interval=interval, blit=False
    )
    return ani