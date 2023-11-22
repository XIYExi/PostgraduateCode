import torch
import torch.nn as nn
import torch.nn.functional as F


class SGC(nn.Module):
    def __init__(self, input_feature, output_feature, k):
        super(SGC, self).__init__()
        self.input_feature = input_feature
        self.output_feature = output_feature
        self.w = nn.Parameter(torch.FloatTensor(input_feature, output_feature))
        nn.init.xavier_uniform_(self.w, gain=1)
        self.k = k

    def forward(self, h, adj):
        output = adj
        for i in range(self.k):
            output = torch.mm(output, adj)
        output = torch.mm(output, h)
        output = torch.mm(output, self.w)
        return F.log_softmax(output, dim=1)
