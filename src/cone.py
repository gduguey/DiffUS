import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
import cv2

def voxel_to_world(idx_ijk: np.ndarray, affine: np.ndarray) -> np.ndarray:
    ijk1 = np.concatenate((idx_ijk, [1.0]))
    xyz1 = affine.dot(ijk1)
    return xyz1[:3]

def world_to_voxel(xyz: np.ndarray, affine: np.ndarray) -> np.ndarray:
    inv_aff = np.linalg.inv(affine)
    xyz1 = np.concatenate((xyz, [1.0]))
    ijk1 = inv_aff.dot(xyz1)
    return ijk1[:3]

def mri_to_us_point(i_mri: int,
                      j_mri: int,
                      slice_idx: int,
                      T1_vol: np.ndarray,
                      T1_affine: np.ndarray,
                      US_vol: np.ndarray,
                      US_affine: np.ndarray):
    D_t1, H_t1, W_t1 = T1_vol.shape
    if not (0 <= slice_idx < W_t1 and 0 <= i_mri < D_t1 and 0 <= j_mri < H_t1):
        raise ValueError(f"T1 : indices are out of range (i={i_mri}, j={j_mri}, k={slice_idx})")
    mri_idx_3d = np.array([i_mri, j_mri, slice_idx])
    world_pt = voxel_to_world(mri_idx_3d, T1_affine)
    us_idx_f = world_to_voxel(world_pt, US_affine)
    us_idx = np.round(us_idx_f).astype(int)
    _, _, k_us = us_idx
    us_slice = US_vol[:, :, k_us]  # axial slice in US (H_us × W_us)

    return us_slice, us_idx

def us_to_mri_point(i_us: int,
                    j_us: int,
                    slice_idx: int,
                    US_vol: np.ndarray,
                    US_affine: np.ndarray,
                    T1_vol: np.ndarray,
                    T1_affine: np.ndarray):
    """
    Map a voxel coordinate from US to MRI space and extract the corresponding MRI slice and indices.
    """
    D_us, H_us, W_us = US_vol.shape
    # if not (0 <= slice_idx < D_us and 0 <= i_us < H_us and 0 <= j_us < W_us):
    #     raise ValueError(f"US: indices out of range (i={i_us}, j={j_us}, k={slice_idx})")
    us_idx_3d = np.array([slice_idx, i_us, j_us])
    world_pt = voxel_to_world(us_idx_3d, US_affine)
    mri_idx_f = world_to_voxel(world_pt, T1_affine)
    mri_idx = np.round(mri_idx_f).astype(int)
    k_mri, i_mri, j_mri = mri_idx
    mri_slice = T1_vol[k_mri, :, :]  # axial slice in MRI (H_mri × W_mri)
    return mri_slice, mri_idx

def plot_mri_us_aligned(i_mri: int, j_mri: int, slice_idx: int, T1_vol: np.ndarray, us_slice: np.ndarray, us_idx: np.ndarray):  
    t1_slice = T1_vol[:, :, slice_idx]
    i_us, j_us, k_us = us_idx
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(t1_slice, cmap='gray', origin='lower')
    axes[0].plot(j_mri, i_mri, 'ro', markersize=6)
    axes[0].set_title(f"T1 – slice k={slice_idx}")
    axes[0].axis('off')

    axes[1].imshow(us_slice, cmap='gray', origin='lower')
    axes[1].plot(j_us, i_us, 'ro', markersize=6)
    axes[1].set_title(f"US  – slice k={k_us}")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

def plot_mri_us_aligned_0(i_us: int, j_us: int, slice_idx: int, us_vol: np.ndarray, mri_slice: np.ndarray, mri_idx: np.ndarray):  
    us_slice = us_vol[slice_idx, :, :]
    k_mri, i_mri, j_mri = mri_idx
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(mri_slice, cmap='gray', origin='lower')
    axes[0].plot(i_mri, j_mri, 'ro', markersize=6)
    axes[0].set_title(f"T1 – slice k={k_mri}")
    axes[0].axis('off')

    axes[1].imshow(us_slice, cmap='gray', origin='lower')
    axes[1].plot(i_us, j_us, 'ro', markersize=6)
    axes[1].set_title(f"US  – slice k={slice_idx}")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


def compute_us_apex_and_direction(m_left, b_left, m_right, b_right):
    # Compute intersection point (apex)
    if np.isclose(m_left, m_right):
        raise RuntimeError("The slopes are nearly equal; no defined intersection.")
    x0 = (b_right - b_left) / (m_left - m_right)
    y0 = m_left * x0 + b_left

    # Compute direction vectors pointing INTO the cone from apex
    v_left = np.array([-1, -m_left])  # Left ray: left/down direction
    v_right = np.array([1, m_right])   # Right ray: right/down direction

    # Normalize vectors
    u_left = v_left / np.linalg.norm(v_left)
    u_right = v_right / np.linalg.norm(v_right)

    # Compute opening angle between rays
    dot_product = np.dot(u_left, u_right)
    dot_product = np.clip(dot_product, -1.0, 1.0)  # Avoid numerical issues
    opening_angle = np.arccos(dot_product)

    # Compute bisector direction (mean of unit vectors)
    bisector = u_left + u_right
    bisector /= np.linalg.norm(bisector)  # Normalize

    return {
        "apex": (x0, y0),
        "opening_angle": opening_angle,
        "direction_vector": bisector
    }

def plot_us_with_affine_lines(us_slice: np.ndarray,
                              m_left: float, b_left: float,
                              m_right: float, b_right: float):
    _, W = us_slice.shape
    plt.figure(figsize=(6, 6))
    plt.imshow(us_slice, cmap="gray", origin="lower")
    plt.title("US slice with affine lines to adjust")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    
    x_vals = np.array([0, W-1])
    plt.plot(x_vals, m_left*x_vals + b_left, 'c--', linewidth=2)
    plt.plot(x_vals, m_right*x_vals + b_right, 'm--', linewidth=2)
    plt.axis('off')
    plt.show()

def overlay_cone(us_slice: np.ndarray,
                    apex: np.ndarray,
                    direction_vector: np.ndarray,
                    opening_angle: float):
    H, W = us_slice.shape
    x0, y0 = apex
    
    # Create coordinate grid
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    
    # Vector from apex to each point
    vx = xx - x0
    vy = yy - y0
    norm_v = np.sqrt(vx**2 + vy**2) + 1e-8  # Avoid division by zero
    
    # Unit vectors
    ux = vx / norm_v
    uy = vy / norm_v
    
    # Dot product with cone direction
    dx, dy = direction_vector
    dot = ux*dx + uy*dy
    
    # Cone mask (points within half-angle of bisector)
    half_angle = opening_angle / 2.0
    mask_cone = dot >= np.cos(half_angle)
    
    return mask_cone

def plot_overlay_cone(us_slice: np.ndarray, mask_cone: np.ndarray, ax=None, title="US slice with cone overlay"):
    H, W = us_slice.shape
    overlay = np.zeros((H, W, 4), dtype=float)
    overlay[..., 0] = 1  # Red channel
    overlay[..., 3] = mask_cone * 0.3  # Alpha channel (30% opacity)
    if ax is None:
        plt.figure(figsize=(6, 6))
        ax = plt.gca()
    ax.imshow(us_slice, cmap="gray", origin="lower")
    ax.imshow(overlay, origin="lower")
    ax.set_title(title)
    ax.axis('off')

def cone_us_to_mri_world(
        apex_us_vox,           # (x, y, z) in US voxel coordinates
        direction_vec_us_2d,   # (dx, dy) in US voxel space (2D)
        US_affine,             # 4x4
        T1_affine              # 4x4
    ):

    # Transform apex from US voxel to MRI voxel coordinates
    apex_world = voxel_to_world(apex_us_vox, US_affine)  # Convert to world coordinates
    apex_t1_vox = world_to_voxel(apex_world, T1_affine)  # Convert to MRI voxel coordinates

    # Transform direction vector (rotation only, no translation)
    # Extract rotation matrices from affines (3x3 top-left)
    R_us = US_affine[:3, :3]
    R_t1 = T1_affine[:3, :3]

    # Apply rotation transformations: MRI_dir = (R_t1 @ inv(R_us)) @ US_dir
    direction_vec_3d = np.append(direction_vec_us_2d, 0)  # Convert 2D to 3D (z=0)
    rotated_dir = np.linalg.inv(R_us) @ direction_vec_3d  # Undo US rotation
    direction_vec_t1 = R_t1 @ rotated_dir  # Apply MRI rotation
    direction_vec_t1 = direction_vec_t1[:2] / np.linalg.norm(direction_vec_t1[:2])  # Normalize 2D

    return apex_t1_vox, direction_vec_t1

def plot_median_line(us_slice, apex, direction_vector, d1, d2, ax=None):
    if ax is None:
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
    x0, y0 = apex
    dx, dy = direction_vector
    
    # Calculate segment endpoints
    p1 = (x0 + d1 * dx, y0 + d1 * dy)
    p2 = (x0 + d2 * dx, y0 + d2 * dy)
    
    ax.imshow(us_slice, cmap='gray', origin='lower')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    # Plot full median line (dashed)
    ax.axline((x0, y0), slope=dy/dx if dx != 0 else 1e10, 
              color='cyan', linestyle='--', alpha=0.5)
    
    # Plot selected segment
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
            'r-', linewidth=3, label=f'd1={d1}, d2={d2}')
    
    # Mark apex and endpoints
    ax.scatter(p1[0], p1[1], s=80, c='lime', marker='o', label='Start')
    ax.scatter(p2[0], p2[1], s=80, c='red', marker='o', label='End')
    
    ax.set_title("Ultrasound Median Line")
    ax.legend()
    ax.axis('off')

