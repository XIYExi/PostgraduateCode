# [2023] DreamRec

> Generate What You Prefer: Reshaping Sequential Recommendation via Guided Diffusion(中科大)


代码：https://github.com/YangZhengyi98/DreamRec

推荐模型范式为：给定一个正项，推荐模型执行负采样以添加负项，并根据用户的历史交互序列学习对用户是否更喜欢它们进行分类。

但是上述范式，有两个局限：
1. it may differ from human behavior
2. the classification is limited in the candidate pool with noisy or easy supervision

DreamRec也用了一个Transformer架构来生成样本，去噪中生成了一个oracle item来恢复正样，从而砍掉负样本，直接描述用户的真实偏好
