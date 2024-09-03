# [2019]BERT4Rec

> BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer(阿里)


本文提出传统的 从左到右的单项模型具有以下局限：

1. 单项架构限制了用户行为行为序列中隐藏表示二点能力
2. they often assume a rigidly ordered sequence which is not always practical

BERT4Rec采用 深度双向自注意 来建模用户行为序列。本文学习了一个双向表示的模型，通过允许用户历史行为中的每个项目融合左右两侧的信息来进行推荐。

```
For example, one may purchase accessories (e.g., Joy-Con controllers) soon after buying a Nintendo Switch, though she/he will not buy console accessories under normal circumstances
```

上述是文中提出的一个案例，但是这也是我觉得问题的所在：
用来进行SRec的文章都存在一个逻辑上的误区，即 ```我购买了 Switch 所以我要买 Joy-Con 手柄```， 因为这是具有连续属性的，在直觉上也是成立的， ```我买了游戏界 - （缺手柄）买手柄```.
但是我认为存在的误区为： ```我购买Joy-Con是因为我买了Switch游戏机，我不买Xbox或者PS5手柄是因为Switch用不了！```

所以我认为在一组用户mask的学习中，后置项应该占有更高的权重。


BERT中通过完形填空的方式进行training，如[mask]掉一部分交互item，然后通过上下文预测当前被mask掉的item id。
最后为了解决训练和预测任务不一致的问题，本文在结尾添加了一个[mask]来进行预测。

