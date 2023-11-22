import torch
from torch import nn
import torch.nn.functional as F

from layers import GATLayer


class GAT(nn.Module):
    def __init__(self, features, hidden, classes, heads, dropout):
        super(GAT, self).__init__()
        self.layers = [GATLayer(features, hidden) for _ in range(heads)]
        for index, layer in enumerate(self.layers):
            self.add_module('layer_{}'.format(index), layer)
        self.last_layer = GATLayer(heads * hidden, classes)
        self.heads = heads
        self.dropout = dropout

    def forward(self, h, adj):
        output = torch.cat([layer(h, adj) for layer in self.layers], dim=1)
        if self.dropout is not None:
            output = F.dropout(output, training=self.training)
        output = self.last_layer(output, adj)
        return F.log_softmax(output, dim=1)
