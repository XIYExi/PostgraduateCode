import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from model.modules import Encoder, LayerNorm, XNetLoss, XNetLossCrossView,  InfoNCE_Linear


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
                 linear_infonce=False, initializer_range=0.02, num_hidden_layers=1
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

        self.conditional_encoder = Encoder(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            hidden_dropout_prob=hidden_dropout_prob,
            hidden_act=hidden_act,
            num_hidden_layers=num_hidden_layers
        )


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


    def add_position_embedding(self, input_ids, frequency_items_mask):
        # 不动原始序列，copy一份seq用来处理 positional 信息
        sequence = input_ids  # (128, 25)
        # (128, 25) 有数据的位置为1， 其他位置为0
        attention_mask = (input_ids > 0).long()

        # 融合 items 频率信息
        attention_frequency_mask = torch.zeros_like(attention_mask, dtype=torch.long)
        for i in range(frequency_items_mask.size(0)):
            for j in range(frequency_items_mask.size(1)):
                if frequency_items_mask[i, j].item() == 0:
                    attention_frequency_mask[i, j] = 1

        assert attention_mask.shape == frequency_items_mask.shape == attention_frequency_mask.shape

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64 (128, 1, 1, 25)
        max_len = attention_mask.size(-1)  # 25
        attn_shape = (1, max_len, max_len)  # (1, 25, 25) 注意力机制上三角掩码

        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # 上三角mask，对角线都是0
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        # 确保生成的seq只依赖左侧的数据，保证model不会作弊
        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        # (128, 1, 25, 25)
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0



        # 仿照 position emb 处理方法，同样处理一份frequency emb
        extended_frequency_attention_mask = attention_frequency_mask.unsqueeze(1).unsqueeze(2)
        attn_frequency_shape = (1, max_len, max_len)
        subsequent_fre_mask = torch.triu(torch.ones(attn_frequency_shape), diagonal=1)
        subsequent_fre_mask = (subsequent_fre_mask == 0).unsqueeze(1)
        subsequent_fre_mask = subsequent_fre_mask.long()
        if self.args.cuda_condition:
            subsequent_fre_mask = subsequent_fre_mask.cuda()
        extended_frequency_attention_mask = extended_frequency_attention_mask * subsequent_fre_mask
        extended_frequency_attention_mask = extended_frequency_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_frequency_attention_mask = (1.0 - extended_frequency_attention_mask) * -10000.0

        seq_length = sequence.size(1)

        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)  # (128, 20)
        item_embeddings = self.item_embeddings(sequence)              # (128, 20, 128)
        position_embeddings = self.position_embeddings(position_ids)  # (128, 20, 128)
        sequence_emb = item_embeddings + position_embeddings          # (128, 20, 128)
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb, extended_attention_mask



    def forward(self, input_ids, target_pos, target_neg, aug_input_ids, epoch, frequency_items_mask):
        # 添加 positional embedding
        input_emb, extended_attention_mask = self.add_position_embedding(input_ids=input_ids, frequency_items_mask=frequency_items_mask)
        aug_input_emb, aug_extended_attention_mask = self.add_position_embedding(aug_input_ids, frequency_items_mask)

        conditional_emb = self.conditional_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)[-1]