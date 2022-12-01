# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import math
from torch import arange, exp, zeros, sin, cos, swapaxes, Tensor
from mmcv.cnn import normal_init

from ..builder import HEADS
from .base import AvgConsensus, BaseHead

# Original source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = arange(max_len).unsqueeze(1)
        div_term = exp(arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = sin(position * div_term)
        pe[:, 0, 1::2] = cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor, batch_first = False) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim] if batch_first = False, [batch_size, seq_len, embedding_dim] if batch_first = True
        """
        if batch_first == False:
            x = x + self.pe[:x.size(0)]
        else:
            x = swapaxes(x, 0, 1)
            x = x + self.pe[:x.size(0)]
            x = swapaxes(x, 0, 1)
        return self.dropout(x)

@HEADS.register_module()
class TLFPosEncHead(BaseHead):
    """Transformer Late-Fusion head with an additional positional encoding.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss').
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        consensus (dict): Consensus config dict.
        dropout_ratio (float): Probability of dropout layer. Default: 0.4.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 consensus=dict(type='AvgConsensus', dim=1),
                 dropout_ratio=0.4,
                 init_std=0.01,
                 t_layers=1,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls=loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.t_layers = t_layers

        consensus_ = consensus.copy()

        consensus_type = consensus_.pop('type')
        if consensus_type == 'AvgConsensus':
            self.consensus = AvgConsensus(**consensus_)
        else:
            self.consensus = None

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool2d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avg_pool = None

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.in_channels, nhead=8, dim_feedforward=self.in_channels, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.t_layers)

        self.pos_enc = PositionalEncoding(d_model=self.in_channels)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x, num_segs):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.
            num_segs (int): Number of segments into which a video
                is divided.
        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(2)
            x = x.unsqueeze(3)

        # [N * num_segs, in_channels, 7, 7]
        if self.avg_pool is not None:
            if isinstance(x, tuple):
                shapes = [y.shape for y in x]
                assert 1 == 0, f'x is tuple {shapes}'
            x = self.avg_pool(x)
            # [N * num_segs, in_channels, 1, 1]
        x = x.reshape((-1, num_segs) + x.shape[1:])
        # [N, num_segs, in_channels, 1, 1]
        x = x.squeeze(4)
        x = x.squeeze(3)
        # [N, num_segs, in_channels]
        x = self.pos_enc(x, batch_first=True)
        # [N, num_segs, in_channels]
        x = self.transformer_encoder(src=x)
        # [N, num_segs, in_channels]
        x = self.consensus(x)
        # [N, 1, in_channels]
        x = x.squeeze(1)
        # [N, in_channels]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score
