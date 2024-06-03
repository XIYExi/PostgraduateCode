# [2023]DiffRec

> Diffusion Recommender Model(新加坡国立)
>
> 第一篇将diffusion用于推荐的论文


Diff-Rec提出了两个变体：
1. ：L-DiffRec 对item进行聚类以进行维度压缩，并在潜在空间中进行扩散过程 
2. T-DiffRec 根据交互时间戳重新加权用户交互以编码时间信息

## Generative recommender

生成推荐模型主要分为两类：

### GAN-based models

utilize a generator to estimate users' interaction probabilities and leverage adversarial training to optimize the parameters 

对抗性训练通常不稳定，导致性能不理想

### VAEs-based models

use an encoder to approximate the posterior distribution over latent factors and maximize the likelihood of observed interactions

但 VAE 却面临着易处理性和表示能力之间的权衡问题。易于处理且简单的编码器可能无法很好地捕获异质用户偏好，而复杂模型的后验分布可能很棘手


![](./figure-1.png)


DiffRec 通过在前向过程中注入预定的高斯噪声来逐渐破坏用户的交互历史，然后通过参数化神经网络迭代地从损坏的交互中恢复原始交互。

L-DiffRec 将项目聚类成组，通过组特定的VAE将每个组上的交互向量压缩为低维潜在向量，并在潜在空间中进行正向和反向扩散过程。L-DiffRec显着减少了模型参数和内存成本，增强了大规模项目预测的能力。

T-DiffRec 通过简单而有效的时间感知重新加权策略对交互序列进行建模。直观上，用户后来的交互被分配更大的权重，然后输入 DiffRec 进行训练和推理。



## DiffRec Training

![](image1.png)

the prior matching term被忽略，因为是一个常数

reconstruction term表示从t状态重建t-1状态的概率

因此优化取决于最大化重建项和噪声匹配项




## Diffusion

![](diffusion.png)

前序扩散和反向生成。 

