import numpy as np


# CN相似度
def CN(set1, set2):
    return len(set1 & set2)  # 求交集


def Jaccard(set1, set2):
    return CN(set1, set2) / len(set1 | set2)


# 两个向量间的cos相似度
def cos4vector(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# 两个集合间的cos相似度
def cos4set(set1, set2):
    return len(set1 & set2) / (len(set1) * len(set2)) ** 0.5


# 两个向量间pearson相似度
def Pearson(v1, v2):
    v1_mean = np.mean(v1)
    v2_mean = np.mean(v2)
    return ((v1 - v1_mean)*(v2-v2_mean)) / (np.linalg.norm(v1-v1_mean) * np.linalg.norm(v2-v2_mean))

# 通过cos4vector调用pearson
def PearsonSet(v1, v2):
    v1 -= np.mean(v1)
    v2 -= np.mean(v2)
    return cos4vector(v1, v2)


if __name__ == '__main__':
    a = [1, 3, 2]
    b = [8, 9, 1]

    print( cos4vector( a, b ) )
    print( Pearson( a, b ) )
    print( PearsonSet( a, b ) )

