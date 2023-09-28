import numpy as np


def svd(data, k):
    u, i, v = np.linalg.svd(data)
    u = u[:, 0:k]
    i = np.diag(i[0:k])
    v = v[0:k, :]
    return u, i, v


def predictSingle(u_index, i_index, u, i, v):
    return u[u_index].dot(i).dot(v.T[i_index].T)


def play():
    k = 3
    # 假设用户物品共现矩阵如下
    data = np.mat([[1, 2, 3, 1, 1],
                   [1, 3, 2.1, 1, 2],
                   [3, 1, 1, 2, 1],
                   [1, 2, 3, 3, 1]])
    u, i, v = svd(data, k)
    print(u.dot(i).dot(v))

    print(predictSingle(2, 1, u, i, v))


if __name__ == '__main__':
    play()
