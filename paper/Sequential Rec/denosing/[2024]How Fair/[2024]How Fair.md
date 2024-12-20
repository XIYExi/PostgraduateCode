# [2024] How Fair is Your Diffusion Recommender Model?

扩散模型在学习数据样本分布时可能会无意中携带信息偏差并导致不公平的结果。 如DiffRec，
利用**扩散模型可以有效地学习去噪用户包含噪声交互的隐式反馈**，
用于解决传统的生成 RS 的有限表示能力（前者）或不稳定的训练过程（后者）。

但是在ML领域，一些作品提出扩散模型可能会在这种有偏差的数据中学习到不公平的结果。需要注意，论文中提到：
**事实上，值得强调的是，此类推荐模型（基于Diffusion）仍处于对其性能的彻底研究阶段**，意思是Diffusion和SR结合领域仍然空间巨大，24年10月位置，
开源出来的Diffusion领域推荐文章仅不足30篇！序列推荐领域能点出来的只有（DiffuRec，DiffCLRec，CDDRec，DiffRec（苏大），Diff4Rec，PDRec），
其中还有好多没开源！

基于Diff的推荐模型中，用户的历史交互不会被破坏为纯噪声，但添加噪声的比例会持续下降。

后面懒得看了，总之就是说 L-DiffRec这个模型好。

