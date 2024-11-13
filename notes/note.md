
## 数据预处理

用户交互序列```history interaction sequence```$H_0$

```target item``` $e_0$

首先通过```GRU+SA```获得提取之后的用户兴趣模块```G_0```


参照DiffuRec和DCRec的方案，合并最终的用户交互矩阵为 $Z_0$:

$$
Z_0 = Concat(G_0, e_0)
$$

## 加噪

使用传统的高斯噪声：

$$
q(Z_t|Z_{t-1}) = \sqrt{\alpha_t}Z_0 + \sqrt{1-\alpha_t}Z_{t-1}
$$

那么现在，进入去噪阶段我们一共有个数据：

$$
原始交互\quad H_0 \\
增强交互\quad G_0 \\
时间T(去噪阶段) \quad t \\
input \quad Z_0 = Concat(G_0, e_0) \\
当前交互\quad Z_t \\
$$




