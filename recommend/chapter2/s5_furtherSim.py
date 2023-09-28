# 通过导入流行度机制去衰减热门产品的相似权重
import math


def getPopularity(data_sets):
    p = dict()
    for id in data_sets:
        frequency = len(data_sets[id])
        ppl = math.log1p(frequency)  # ln(1+x)
        p[id] = ppl
    return p


def getIIFSim(s1, s2, popularities):
    """
    :param s1: 用户或物品集合 {iid1,iid2}
    :param s2: 用户或物品集合 {iid2,iid3}
    :param popularities: 流行度字典 {iid1: ppl1, iid2:ppl2}
    :return: IIF相似度
    """
    s = 0
    for i in s1 & s2:
        s += 1 / popularities[i]
    return s / (len(s1) * len(s2)) ** .5


# 归一化
def normalizePopularities(popularities):
    """
    :param popularities: 流行度字典 {iid1: ppl1, iid2:ppl2}
    :return: 归一化后的流行度字典 {iid1: ppl1, iid2:ppl2}
    """
    maxp = max(popularities.values())
    norm_ppl = dict()
    for i in popularities:
        norm_ppl[i] = popularities[i] / maxp
    return norm_ppl


# alpha相似度
def getAlphaSim(s1, s2, norm_ppl1):
    """
    :param s1: 用户或物品集合 {iid1,iid2}
    :param s2: 用户或物品集合 {iid2,iid3}
    :param norm_ppl1: 归一化后的流行度字典 {iid1: ppl1, iid2:ppl2}
    :return: alpha相似度
    """
    alpha = (1 + norm_ppl1) / 2
    return len(s1 & s2) / (len(s1) ** (1 - alpha) * len(s2) ** alpha)


def Sigmoid(x):
    return 1 / (1 + math.e ** (-x))


if __name__ == '__main__':
    print(2 ** 0.2)
    print(2 ** 0.8)
    print(Sigmoid(-6))
