# [2024] PDRec

> Plug-in Diffusion Model for Sequential Recommendation(山大、腾讯)
> https://github.com/hulkima/PDRec


本文的思路为：通用的推荐使用语料库中得分最高的item来进行用户兴趣预测，这会导致忽略其余item中包含的用户广义偏好.


具体步骤中PDRec先通过time-interval diffusion model来生成用户对所有item的偏好，并且通过融合历史行为重加权（HBR）来识别高质量行为。

同时论文还提出了一种基于diffusion的正增强策略DPA，利用排名靠前但是未被观测到的项目进行潜在正样本，引入信息丰富并且多样化的软信号来缓解数据稀疏性。


## Introduction

文章提出DiffRec等模型根据user 历史交互数据进行扩散，但是存在两个挑战：
1. 如何充分利用DM的广义用户偏好
   
   如diffrec等模型，只是输出了语料库中得分高的item，但是却忽略了用户对于其余item的偏好。

2. 如何基于Diffusion的知识构建一个可以与不同SR Model写作的通用框架？

    【这是本文提出的创新点，也就是提出了插件化序列推荐Diff-based model，但是在同年港大的DiffCLRec中也对这个问题进行了回答】

    ***可以提出一个新的解决思路？使用plug-in diffusion model去提取广义的用户偏好，然后使用港大的思路集合上下文（BERT）和HBR生成私有用户的个人偏好，结合（cat）之后进行CL training***


