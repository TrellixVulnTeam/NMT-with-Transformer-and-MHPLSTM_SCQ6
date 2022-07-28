import torch
import torch.nn as nn
from HPLSTM import MultiHeadHPLSTMModule1, MultiHeadHPLSTMModule2
from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward


class MHPLSTMDecoderLayer(nn.Module):
    """Replace decoder self attention network with MHPLSTM"""
    