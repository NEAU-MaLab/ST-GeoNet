"""Trajectory loading, graph normalization, masked loss, and logging utilities."""

import os
import json
from pathlib import Path
import torch
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from trajectory_schema import get_extra_columns, make_feature_config, read_trajectory_csv

def load_checkpoint(model, path, device):
    """Load model weights strictly and check any accompanying feature configuration.

    New checkpoints have a .features.json sidecar recording feature names and
    their order. A checkpoint without that sidecar uses the default six inputs.
    """
    path = Path(path)
    target = model.module if isinstance(model, torch.nn.DataParallel) else model
    expected = getattr(target, 'feature_config', None)
    config_path = Path(str(path) + '.features.json')
    if expected is not None:
        if config_path.exists():
            stored = json.loads(config_path.read_text(encoding='utf-8'))
            if stored != expected:
                raise ValueError(
                    f'{config_path}: feature configuration does not match this model. '
                    'Use the same --extra_features order and --motion_source as training.')
        elif expected != make_feature_config():
            raise ValueError(
                f'{path}: a matching .features.json file is required for extra features '
                'or CSV motion inputs. Copy it together with the trained checkpoint.')
    state_dict = torch.load(path, map_location=device, weights_only=True)
    consume_prefix_in_state_dict_if_present(state_dict, 'module.')
    return target.load_state_dict(state_dict, strict=True)

def save_checkpoint(model, path):
    """Save a state dictionary and its ordered input-feature configuration."""
    path = Path(path)
    torch.save(model.state_dict(), path)
    target = model.module if isinstance(model, torch.nn.DataParallel) else model
    config = getattr(target, 'feature_config', None)
    if config is not None:
        config_path = Path(str(path) + '.features.json')
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

def cal_loss(pred, gold, ignore_index=-1):
    """Compute mean cross-entropy from (B*N, 2) logits, ignoring padded labels."""

    gold = gold.contiguous().view(-1)

    loss = F.cross_entropy(
        pred, gold, reduction='mean',
        ignore_index=ignore_index)

    return loss

class IOStream:
    """Write each progress message to both the console and an append-only log."""
    def __init__(self, path):
        """Open the experiment log in append mode."""
        self.f = open(path, 'a')

    def cprint(self, text):
        """Print a message and flush it to the experiment log."""
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        """Close the log file."""
        self.f.close()

def pad_data(data, target_size=4096, pad_value=0):
    """Truncate or pad the first axis; multidimensional arrays are zero-padded."""
    current_size = data.shape[0]
    if current_size >= target_size:
        return data[:target_size]

    if len(data.shape) == 1:
        padded_data = np.full(target_size, pad_value, dtype=data.dtype)
        padded_data[:current_size] = data
    else:
        padded_data = np.zeros((target_size, *data.shape[1:]), dtype=data.dtype)
        padded_data[:current_size] = data
    return padded_data

def normalize2(arr):
    """Min-max scale one attribute; near-constant columns become zeros."""
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val - min_val < 1e-6:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)

def load_data(path="data", exp_mode='train', target_size=4096, extra_features=None, motion_source='derived'):
    """Load paired Feature/Graph files into data, labels, masks, and adjacency.

    Returns tensors with shapes (T, N, 6+K), (T, N), (T, N), and (T, N, N),
    where T counts trajectories, N is target_size, and K counts selected extra
    features. Input order is x, y, zero, two motion attributes, extras, degree.
    Motion attributes and extras are independently min-max scaled per trajectory.
    motion_source='derived' computes forward distance/angle from coordinates;
    motion_source='csv' uses the named speed/direction columns instead.
    """
    path = str(path).strip("'\"")
    config = make_feature_config(extra_features, motion_source)
    extra_features = config['extra_features']
    Adj, Features, Labels, Masks = [], [], [], []
    feature_path = os.path.join(path, exp_mode, 'Feature')
    graph_path = os.path.join(path, exp_mode, 'Graph')

    for features_name in os.listdir(feature_path):
        graph_name = features_name.replace('feature', 'graph')
        csv_path = os.path.join(feature_path, features_name)
        frame = read_trajectory_csv(csv_path, prepared=True)
        available_extras = get_extra_columns(frame.columns, prepared=True)
        missing = [name for name in extra_features if name not in available_extras]
        if missing:
            raise ValueError(f'{csv_path}: missing requested extra feature columns: {missing}.')
        features = np.column_stack((frame.index.to_numpy(), frame.to_numpy())).astype(np.float32)
        labels = features[:, -1]  # Read the final CSV column as the point label (0 or 1).
        original_size = features.shape[0]
        if not 2 <= original_size <= target_size:
            raise ValueError(f'{csv_path}: expected 2 to {target_size} points, got {original_size}.')
        xy = features[:, 1:3]  # Skip the saved CSV index and select the coordinate columns.
        if motion_source == 'derived':
            deltas = xy[1:] - xy[:-1]
            distances = np.linalg.norm(deltas, axis=1)
            angles = np.arctan2(deltas[:, 1], deltas[:, 0])
            # Reuse the final forward difference at the last trajectory point.
            distances = np.append(distances, distances[-1])
            angles = np.append(angles, angles[-1])
        else:
            distances = frame['speed'].to_numpy(dtype=np.float32)
            angles = frame['direction'].to_numpy(dtype=np.float32)
        if not np.isfinite(distances).all() or not np.isfinite(angles).all():
            raise ValueError(f'{csv_path}: motion features must contain finite numeric values.')

        # Select extras by name and CLI order, independently of their CSV column positions.
        extras = np.empty((original_size, len(extra_features)), dtype=np.float32)
        for column, name in enumerate(extra_features):
            values = frame[name].to_numpy(dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError(f'{csv_path}: {name} contains NaN or infinite values.')
            extras[:, column] = normalize2(values)
        extras = pad_data(extras, target_size)

        # Normalize the two selected motion attributes independently per trajectory.
        distances = normalize2(distances)
        angles = normalize2(angles)

        # Insert the selected motion columns after the saved index and coordinates.
        new_features = np.concatenate([
            features[:, :3],  # Saved index, x, and y.
            distances[:, None],
            angles[:, None],
            features[:, 3:]  # Preserve the remaining source columns for their existing column positions.
        ], axis=1)

        features = new_features

        # Mark real points with 1 and padding positions with 0.
        mask = np.zeros(target_size, dtype=np.float32)
        mask[:original_size] = 1.0

        # Pad to the requested point count; labels use -1 to exclude padding from loss.
        features = pad_data(features, target_size)
        labels = pad_data(labels, target_size, pad_value=-1)  # Ignore padding labels during loss and metric calculations.

        # Map graph node IDs to row indices in the feature table.
        idx = np.array(features[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx[:original_size])}

        edges = np.loadtxt(os.path.join(graph_path, graph_name), delimiter=',', dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges.flatten())), dtype=np.int32).reshape(edges.shape)

        # Symmetrize adjacency, add self-loops, and apply row normalization.
        adj = sp.coo_matrix((
            np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            shape=(original_size, original_size), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = normalize(adj + sp.eye(adj.shape[0]))

        # Extend the CSR row pointer; padded points have no graph connections.
        new_indptr = np.zeros(target_size + 1, dtype=adj.indptr.dtype)
        new_indptr[:original_size + 1] = adj.indptr  # Keep the row pointers of real points.
        new_indptr[original_size + 1:] = adj.indptr[-1]  # Empty rows for padding.
        padded_adj = sp.csr_matrix(
            (adj.data, adj.indices, new_indptr),
            shape=(target_size, target_size),
            dtype=np.float32)

        # Coordinates remain first; selected extras follow motion, and degree stays last.
        features = np.concatenate([
            features[:, [1, 2]], np.zeros((features.shape[0], 1)),
            features[:, [3, 4]], extras, features[:, [-2]],
        ], axis=1)
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels)
        adj = sparse_mx_to_torch_sparse_tensor(padded_adj)
        mask = torch.FloatTensor(mask)

        Adj.append(adj)
        Features.append(features)
        Labels.append(labels)
        Masks.append(mask)

    # Load all trajectories into memory, including dense padded adjacency matrices.
    return torch.stack(Features), torch.stack(Labels), torch.stack(Masks), torch.stack(Adj).to_dense()

def normalize(mx):
    """Return D^-1 times a sparse matrix, with zero rows left at zero."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(mx)

def accuracy(output, labels, mask=None):
    """Compute point accuracy from scores (B, N, C) and labels (B, N).

    If supplied, mask has shape (B, N), with 1 for real points and 0 for padding.
    Otherwise, labels equal to -1 are excluded. Scores need not be probabilities.
    """
    preds = output.argmax(dim=-1)  # (batch_size, num_nodes)
    correct = preds.eq(labels).float()  # Convert boolean correctness indicators to floating-point values.

    if mask is not None:
        # Count only the real points selected by the supplied mask.
        correct = correct * mask
        return correct.sum() / mask.sum()

    # Without an explicit mask, ignore padding labels (-1).
    valid_mask = labels.ne(-1)  # Select non-padding labels.
    return correct[valid_mask].sum() / valid_mask.sum()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a SciPy sparse matrix into a float32 PyTorch COO tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
