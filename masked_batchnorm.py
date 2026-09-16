"""Batch-normalization wrappers with optional point masks."""

import torch
import torch.nn as nn

class MaskedBatchNorm1d(nn.Module):
    """Wrap BatchNorm1d with an optional masked training path.

    Without a mask, or during evaluation, the registered bn module is used.
    The masked path computes statistics over real points in the current
    batch without updating running statistics.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        """Create the underlying normalization module and its state buffers."""
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=True, track_running_stats=True)

    def forward(self, x, mask=None):
        """Normalize (B, C, N) features using an optional (B, N) point mask.

        The unmasked path also supports the input shapes accepted by BatchNorm1d.
        """
        # Use standard BatchNorm when no training mask is supplied.
        if mask is None or not self.training:
            return self.bn(x)

        # Flatten batch and point dimensions to compute channel-wise statistics.
        B, C, N = x.shape
        x = x.permute(0, 2, 1).reshape(B * N, C)
        mask = mask.reshape(B * N).float().unsqueeze(1)

        # Compute masked statistics without updating the registered running buffers.
        x_mean = (x * mask).sum(dim=0) / (mask.sum() + 1e-5)
        x_var = ((x - x_mean) ** 2 * mask).sum(dim=0) / (mask.sum() + 1e-5)

        # Apply epsilon and affine parameters; padded outputs are not zeroed.
        x = (x - x_mean) / torch.sqrt(x_var + 1e-5)
        x = x * self.bn.weight + self.bn.bias
        x = x.view(B, N, C).permute(0, 2, 1)
        return x

class MaskedBatchNorm2d(nn.Module):
    """Wrap BatchNorm2d with an optional masked training path.

    The point mask is broadcast over neighbors. The unmasked and evaluation
    paths use the registered BatchNorm2d module directly.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        """Create the underlying normalization module and its state buffers."""
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=True, track_running_stats=True)

    def forward(self, x, mask=None):
        """Normalize (B, C, N, k) edge features with an optional (B, N) mask."""
        # Use standard BatchNorm when no training mask is supplied.
        if mask is None or not self.training:
            return self.bn(x)

        # Flatten batch, point, and neighbor dimensions.
        B, C, N, k = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B * N * k, C)
        mask = mask.unsqueeze(-1).expand(B, N, k).reshape(B * N * k, 1).float()

        # Compute masked statistics without updating the registered running buffers.
        x_mean = (x * mask).sum(dim=0) / (mask.sum() + 1e-5)
        x_var = ((x - x_mean) ** 2 * mask).sum(dim=0) / (mask.sum() + 1e-5)

        # Apply epsilon and affine parameters; padded outputs are not zeroed.
        x = (x - x_mean) / torch.sqrt(x_var + 1e-5)
        x = x * self.bn.weight + self.bn.bias
        x = x.view(B, N, k, C).permute(0, 3, 1, 2)
        return x
