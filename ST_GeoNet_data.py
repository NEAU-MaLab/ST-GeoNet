"""Trajectory dataset and retained point-cloud augmentation helpers.

The augmentation helpers are not called by the current training pipeline.
"""

import numpy as np
import torch
from ST_GeoNet_util import load_data
from torch.utils.data import Dataset

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    """Add clipped Gaussian noise to every channel in place (optional helper)."""
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud

def rotate_pointcloud(pointcloud):
    """Rotate columns 0 and 2 in place using the retained 3-D convention."""
    theta = np.pi * 2 * np.random.uniform()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pointcloud[:, [0, 2]] = pointcloud[:, [0, 2]].dot(rotation_matrix)  # random rotation (x,z)
    return pointcloud

def translate_pointcloud(pointcloud):
    """Apply independent scale and translation to a three-channel point cloud."""
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

class TrajectoryDataset(Dataset):
    """Expose preloaded trajectories with labels, valid-point masks, and graphs."""
    def __init__(self, path='', partition='train', num_points=4096, T_D=False, extra_features=None, motion_source='derived'):
        """Load one data split. T_D retains the original three-channel-only option.

        ST-GeoNet expects 6+K channels, where K is the number of selected extras.
        T_D=True selects three channels and is incompatible with ST-GeoNet.
        """
        self.data, self.seg, self.mask, self.adj = load_data(
            path=path, exp_mode=partition, target_size=num_points,
            extra_features=extra_features, motion_source=motion_source)
        self.num_points = num_points
        self.partition = partition
        if T_D is True:
            self.data = torch.FloatTensor(np.array([np.concatenate([self.data[i][:, 0:2], np.zeros((
                len(self.data[i]), 1))], axis=1) for i in range(len(self.data))]))

    def __getitem__(self, item):
        """Return features (N, 6+K), labels, mask, and adjacency; T_D uses (N, 3)."""
        mask = self.mask[item]
        pointcloud = self.data[item]
        seg = self.seg[item]
        adj = self.adj[item]
        return pointcloud, seg, mask, adj

    def __len__(self):
        """Return the number of trajectories in the selected split."""
        return self.data.shape[0]
