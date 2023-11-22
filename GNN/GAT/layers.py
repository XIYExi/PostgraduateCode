import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter


class GATLayer(nn.Module):
    def __init__(self, input_features, output_features):
        super(GATLayer, self).__init__()
        # 输入特征
        self.input_features = input_features
        # 输出特征
        self.output_features = output_features
        # 权重矩阵
        self.w = Parameter(torch.FloatTensor(input_features, output_features))
        self.a = Parameter(torch.FloatTensor(2 * output_features, 1))
        # 初始化权重
        nn.init.xavier_uniform_(self.w, gain=1.414)
        nn.init.xavier_uniform_(self.a, gain=1.414)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, h, adj):
        # Wh -> [N, output_features]
        Wh = torch.mm(h, self.w)
        # E -> [N, N]
        e = self.get_e(Wh)  # eij
        matrix = -9e15 * torch.ones_like(e)
        # 注意力
        alpha = torch.where(adj > 0, e, matrix)
        alpha = F.softmax(alpha, dim=1)
        output = F.relu(torch.mm(alpha, Wh))
        return output

    def get_e(self, Wh):
        # [Wh1 || Wh2]
        Wh1 = torch.matmul(Wh, self.a[:self.output_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.output_features:, :])
        # e.shape(N, N)
        e = Wh1 + Wh2.T
        return self.leaky_relu(e)
