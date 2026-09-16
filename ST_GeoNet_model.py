"""ST-GeoNet model components for point-wise field-road segmentation.

Author: QiHeng Wu
Contact: s.t_labmem001@foxmail.com
"""

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from masked_batchnorm import MaskedBatchNorm1d, MaskedBatchNorm2d
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from trajectory_schema import make_feature_config

def knn(x, mask=None, k=20):
    """Return k nearest-neighbor indices for features of shape (B, C, N).

    An optional (B, N) mask excludes padded candidate neighbors. The query
    points are retained, and a point may be selected as its own neighbor.
    The result has shape (B, N, k).
    """
    B, C, N = x.size()
    # Compute negative squared Euclidean distances so top-k selects nearest points.
    x = x.transpose(2, 1)
    dist = -2 * torch.matmul(x, x.transpose(2, 1))
    xx = torch.sum(x ** 2, dim=2, keepdim=True)
    dist = -xx - dist - xx.transpose(2, 1)
    if mask is not None:
        mask = mask.bool()
        mask = mask.unsqueeze(1).expand(B, N, N)
        dist[~mask] = float('-inf')
    idx = dist.topk(k=k, dim=-1)[1]
    return idx

def get_graph_feature(x, k=20, idx=None, dim9=False, mask=None):
    """Build EdgeConv edge features with shape (B, 2 * C, N, k).

    Each edge concatenates the neighbor-minus-center difference with the
    center feature. If indices are absent, neighbors are computed from x.
    The legacy dim9 option restricts neighbor search to channels from index 6.
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k, mask=mask)
        else:
            idx = knn(x[:, 6:], k=k, mask=mask)
    device = x.device
    # Offset point indices to gather neighbors across the flattened batch.
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    # Concatenate relative edge features and repeated center features.
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature

class GraphConvolution(nn.Module):
    """Apply a learned feature projection followed by the supplied adjacency.

    Input features have shape (B, N, in_features), and adj has shape
    (B, N, N). Adjacency normalization is performed by the data loader.
    """

    def __init__(self, in_features, out_features, bias=True):
        """Register the projection parameters and initialize their values."""
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights and bias from a uniform distribution."""
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """Return graph features with shape (B, N, out_features)."""

        # Project features first, then aggregate using the precomputed adjacency.
        support = torch.matmul(input, self.weight)
        output = torch.bmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        return output

    def __repr__(self):
        """Show the input and output channel counts in module summaries."""
        return f'{self.__class__.__name__}({self.in_features}->{self.out_features})'

class GCNBackbone(nn.Module):
    """Extract spectral features using three graph convolution layers."""
    def __init__(self, args, in_channels=6, hidden_channels=32):
        """Create three graph convolutions with ReLU and training-time dropout."""
        super(GCNBackbone, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.gc_layers = nn.ModuleList([
            GraphConvolution(in_channels if i == 0 else hidden_channels, hidden_channels)
            for i in range(3)
        ])

    def forward(self, x, adj):
        """Map (B, N, in_channels) features to (B, N, hidden_channels)."""

        for gc in self.gc_layers:
            x = F.relu(gc(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
        return x

class Transform_Net(nn.Module):
    """Predict a 3-by-3 alignment transform for the GeoNet coordinate branch.

    The spatial input uses three coordinate channels.
    """
    def __init__(self, args):
        """Create the spatial alignment network."""
        super(Transform_Net, self).__init__()
        self.args = args
        self.k = 3

        self.bn1 = MaskedBatchNorm2d(64)
        self.bn2 = MaskedBatchNorm2d(128)
        self.bn3 = MaskedBatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        # This attribute normalizes 512 channels; conv3 contains a 1024-channel normalization.
        self.bn3 = MaskedBatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = MaskedBatchNorm1d(256)

        self.transform = nn.Linear(256, 3 * 3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x, mask=None):
        """Map edge features (B, 6, N, k) to transforms (B, 3, 3).

        The mask argument is retained for interface compatibility. This forward
        path does not pass it into its normalization or pooling operations.
        """
        batch_size = x.size(0)

        # Transform edge features and pool over neighbors.
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.max(dim=-1, keepdim=False)[0]

        # Build and pool a global representation for the alignment transform.
        x = self.conv3(x)
        x = x.max(dim=-1, keepdim=False)[0]

        # Predict a transform initialized to the identity matrix.
        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)

        x = self.transform(x)
        x = x.view(batch_size, 3, 3)

        return x

class PositionalEncoding(nn.Module):
    """Add fixed sinusoidal positional encodings with base 5000.

    This implementation is used with even embedding widths. Its persistent
    pe buffer is part of the checkpoint state dictionary.
    """
    def __init__(self, d_model, max_len=4096):
        """Precompute encodings for up to max_len sequence positions."""
        super(PositionalEncoding, self).__init__()
        # Positions index the sequence; alternating channels use sine and cosine.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(5000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encodings to features of shape (B, T, d_model)."""
        return x + self.pe[:, :x.size(1), :]

class Transformer(nn.Module):
    """Extract temporal features through embedding and a Transformer encoder.

    Features are projected before positional encoding is added. The encoder
    uses a feed-forward width of 2048 and ReLU activation.
    """
    def __init__(self, args, input_dim=6, d_model=32, nhead=4, num_layers=3, dropout=0.5, num_classes=2, max_len=4096):
        """Create the embedding, positional encoding, and encoder stack."""
        super(Transformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)

        # The encoder uses a feed-forward width of 2048.
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, mask=None):
        """Return temporal features of shape (B, T, d_model).

        The input has shape (B, T, input_dim). An optional mask of shape
        (B, T) uses 1 for real points and 0 for padding; padded keys are ignored.
        """
        # Embed input channels before adding the 32-dimensional positional encoding.
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        if mask is not None:
            attention_mask = ~mask.bool()
        else:
            attention_mask = None

        # The key-padding mask excludes padded keys from attention.
        x = self.encoder(x, src_key_padding_mask=attention_mask)

        return x

class ST_GeoNet(nn.Module):
    """Segment trajectories with spectral, temporal, and GeoNet branches.

    The input order is [x, y, 0, motion1, motion2, *extra_features, degree].
    The two motion channels follow the configured motion source. With no
    extra features, the six-channel input is retained. Each selected extra
    feature adds one input channel to the spectral and temporal branches.
    GeoNet always uses the first three channels, including the zero channel.
    The output contains two raw class logits per point.
    """
    def __init__(self, args):
        """Create the three branches and point-wise classification decoder."""
        super(ST_GeoNet, self).__init__()
        self.args = args
        self.feature_config = make_feature_config(
            extra_features=getattr(args, 'extra_features', None),
            motion_source=getattr(args, 'motion_source', 'derived'),
        )
        self.input_dim = self.feature_config['input_dim']
        self.k = args.k
        self.graph_feature_dims = args.graph_feature_dims
        # Extra features change only the input widths of these two branches.
        self.gcn_backbone = GCNBackbone(args, in_channels=self.input_dim, hidden_channels=args.graph_feature_dims)

        self.transform_net = Transform_Net(args)
        self.Transformer = Transformer(args, input_dim=self.input_dim)

        # Preserve shared normalization registrations and their checkpoint keys.
        self.bn1 = MaskedBatchNorm2d(64)
        self.bn2 = MaskedBatchNorm2d(64)
        self.bn3 = MaskedBatchNorm2d(64)
        self.bn4 = MaskedBatchNorm2d(64)
        self.bn5 = MaskedBatchNorm2d(64)
        self.bn6 = MaskedBatchNorm1d(args.emb_dims)
        self.bn7 = MaskedBatchNorm1d(64)
        self.bn8 = MaskedBatchNorm1d(256)
        self.bn9 = MaskedBatchNorm1d(256)
        self.bn10 = MaskedBatchNorm1d(128)

        # GeoNet EdgeConv blocks: 6 -> 64 -> 64, 128 -> 64 -> 64, and 128 -> 64.
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        # Project the three local 64-channel outputs into the global embedding.
        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        # This registered layer is not used in forward().
        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        # Decode the fused 1280-channel representation through widths 256, 256, 128, and 2.
        self.conv8 = nn.Sequential(nn.Conv1d(1280, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                    self.bn10,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, 2, kernel_size=1, bias=False)

    def forward(self, x, mask=None, adj=None):
        """Map features (B, N, input_dim) to class logits (B, N, 2).

        input_dim is 6 plus the number of configured extra features. Their
        order matches feature_config['input_order'], with degree last.
        mask has shape (B, N), and adj has shape (B, N, N). Only the temporal
        branch receives mask in this forward path; GeoNet uses the existing
        unmasked neighborhood searches, normalization, and maximum pooling.
        """
        if x.ndim != 3 or x.size(-1) != self.input_dim:
            raise ValueError(
                f'Expected trajectory features with shape (B, N, {self.input_dim}); '
                f'got {tuple(x.shape)}.'
            )

        # Spectral branch: (B, N, input_dim) -> (B, N, graph_feature_dims).
        x_graph = self.gcn_backbone(x, adj)
        x_graph = x_graph.transpose(2, 1)

        # Temporal branch: (B, N, input_dim) -> (B, N, 32).
        x_series_feature = self.Transformer(x, mask)
        x_series_feature = x_series_feature.transpose(2, 1)

        num_points = x.size(1)

        # GeoNet starts from the first three channels, including the zero-valued spatial channel.
        x_spatial = x[:, :, :3].transpose(2, 1)
        # Build alignment-network edges of shape (B, 6, N, k).
        x0 = get_graph_feature(x_spatial, k=self.k)
        t = self.transform_net(x0)

        # Apply the learned 3-by-3 transform to each three-channel spatial input.
        x = x_spatial.transpose(2, 1)
        x = torch.bmm(x, t)
        x = x.transpose(2, 1)

        # First EdgeConv block; neighbor pooling produces (B, 64, N).
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        # Second EdgeConv block; reconstruct neighborhoods from the first local output.
        x = get_graph_feature(x1, k=self.k)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        # Third EdgeConv block; reconstruct neighborhoods from the second local output.
        x = get_graph_feature(x2, k=self.k)
        x = self.conv5(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        # Concatenate local outputs into (B, 192, N).
        x = torch.cat((x1, x2, x3), dim=1)

        # Build a global embedding, then broadcast it to all N input points.
        x = self.conv6(x)
        x = x.max(dim=-1, keepdim=True)[0]
        x = x.repeat(1, 1, num_points)

        # Concatenate global, three local, temporal, and spectral features in that order.
        x = torch.cat((x, x1, x2, x3, x_series_feature, x_graph), dim=1)

        # Apply the point-wise decoder and return logits in point-major order.
        x = self.conv8(x)
        x = self.dp1(x)
        x = self.conv9(x)
        x = self.dp2(x)
        x = self.conv10(x)
        x = self.conv11(x)

        return x.transpose(2, 1)

class Transformer_clear(nn.Module):
    """Provide the retained standalone Transformer segmentation baseline."""
    def __init__(self, args, input_dim=5, d_model=64, nhead=4, num_layers=3, dropout=0.5, num_classes=2, max_len=4096):
        """Create the temporal feature extractor and its two-class output layer."""
        super(Transformer_clear, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)

        # Retain the standalone baseline encoder settings without changing its parameters.
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, mask=None):
        """Return logits of shape (B, T, num_classes) from (B, T, input_dim).

        An optional (B, T) mask marks real points with 1 and padding with 0.
        """
        # Project features, add positional encodings, and ignore padded attention keys.
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        if mask is not None:
            attention_mask = ~mask.bool()
        else:
            attention_mask = None

        # Classify the temporal output independently at every sequence position.
        x = self.encoder(x, src_key_padding_mask=attention_mask)
        logits = self.classifier(x)
        return logits
