# [2024]CALRec

> CALRec: Contrastive Alignment of Generative LLMs For Sequential Recommendation(剑桥,google)


本文使用LLM微调后进行sequence推荐. Despite the impressive language understanding capabilities of pretrained LLMs, they require fine-tuning for highly specific tasks like sequential recommendation.

微调是为了更好的学习到特定领域的扁平化文本序列和潜在的用户兴趣演变。

模型的特性：
1. Pure text input and text output with advanced prompt design
2. A mixed training objective，结合了customized nextitem generation objective和auxiliary contrastive objectives
3. 多类别联合精细组成的两阶段LLM微调范式
4. 一种用于item retrieval的新标准BM25

本文的贡献：

1. 提出了CALRec，一种具有 advanced prompt design, a two-stage training paradigm, a combined training objective, and a quasi-round-robin BM25 retrieval approach 的顺序推荐框架。
2. 达到了sota
3. 重新审视了该领域的数据预处理和评估指标


