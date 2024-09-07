import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from model.modules import LayerNorm


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device) # select the values of different timesteps along the axis pointed by the index of t,
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1)) # reshape out to be (batch_size, 1, 1) if x_shape is (batch_size, h, w)                "")


class DemoRecModel(nn.Module):
    def __init__(self,
                 T, hidden_size, item_size,
                 max_seq_length=20, num_attention_heads=2,
                 attention_probs_dropout_prob=0.2, hidden_act='gelu', hidden_dropout_prob=0.0,
                 linear_infonce=False, initializer_range=0.02
                 ):
        super(DemoRecModel, self).__init__()
        self.item_size = item_size
        self.hidden_size = hidden_size
        self.max_len = max_seq_length
        self.initializer_range = initializer_range

        # 时间embedding，交互序列长度为20，hidden latent大小为128
        self.time_embeddings = nn.Embedding(T, hidden_size)  # [25, 128]

        # TODO conditional encoder

        # [item_num, 128]
        self.item_embeddings = nn.Embedding(self.item_size, self.hidden_size, padding_idx=0)
        # [25, 128]
        self.position_embeddings = nn.Embedding(self.max_len, self.hidden_size)
        self.decoder = nn.TransformerDecoderLayer(
            d_model=self.hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=self.hidden_size,
            dropout=attention_probs_dropout_prob,
            activation=hidden_act
        )
        self.LayerNorm = LayerNorm(self.hidden_size, eps=1e-12)

        self.dropout = nn.Dropout(hidden_dropout_prob)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.BCELoss(reduction='none')

        # TODO 算损失 XNetLoss 或 InfoNCE_Linear
        self.mse = nn.MSELoss()

        self.apply(self.init_weights)

        # diffusion models setting
        # coefficiencets for gaussian diffusion
        self.T = T
        self.beta_1 = 1e-4  #beta_1
        self.beta_T = 0.002 #beta_T
        self.betas = torch.linspace(self.beta_1, self.beta_T, self.T).double().to(self.device)
        self.alphas = 1.0 - self.betas.to(self.device)
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0).to(self.device)
        self.alphas_cumprod_prev = torch.cat((torch.Tensor([1.0]).to(self.device), self.alphas_cumprod[:-1])).to(self.device)
        self.alphas_cumprod_next = torch.cat((self.alphas_cumprod[1:], torch.Tensor([0.0]).to(self.device))).to(self.device)

        # coefficientes for true diffusion distribution q
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(self.device)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod).to(self.device)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod).to(self.device)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1).to(self.device)

        # calculates for posterior distribution q(x_{t-1}|x_t, x_0)
        self.posterior_variance = (
                self.betas * (1 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        ).to(self.device)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat((torch.Tensor([self.posterior_variance[1]]).to(self.device), self.posterior_variance[1:]))
        ).to(self.device)
        self.posterior_mean_coef1 = (
                self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        ).to(self.device)
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        ).to(self.device)


    def init_weights(self, module):
        """
        Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, target_pos, target_neg, aug_input_ids, epoch):
        pass