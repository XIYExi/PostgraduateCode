import torch
from torch import nn
from torch.nn.parameter import Parameter

class GraphConv(nn.Module):
    def __init__(self, input_feature, output_feature):
        super(GraphConv, self).__init__()
        # 输入特征
        self.input_feature = input_feature
        # 输出特征
        self.output_feature = output_feature
        # 权重
        self.weight = Parameter(torch.FloatTensor(input_feature, output_feature))
        # 初始化权重
        nn.init.xavier_uniform_(self.weight, gain=1.4)

    def forward(self, h, adj):
        output = torch.mm(h, self.weight)
        output = torch.mm(adj, output)
        return output

