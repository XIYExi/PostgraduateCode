import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from GAT import GAT
from GNN.GCN.util import load_data, accuracy
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 参数配置
heads = 8
lr = 0.01
hidden = 8
weight_decay = 5e-4
idx_train = range(140)
idx_val = range(200, 500)
idx_test = range(500, 1500)
dropout = 0.6

# 数据集为cora或citeseer
dataset = 'cora'

adj_matrix, features, labels = load_data(dataset)

model = GAT(
    features=features.shape[1],
    hidden=hidden,
    classes=labels.max().item() + 1,
    heads=heads,
    dropout=dropout
)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

if torch.cuda.is_available():
    model = model.cuda()
    features = features.cuda()
    adj_matrix = adj_matrix.cuda()
    labels = labels.cuda()


acc_loss = []
acc = []

val_loss = []
vals = []

def train(epoch):
    model.train()
    optimizer.zero_grad()
    output = model(features, adj_matrix)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])

    acc.append(acc_train.item())
    acc_loss.append(loss_train.item())
    vals.append(acc_val.item())
    val_loss.append(loss_val.item())

    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()))


def val():
    model.eval()
    output = model(features, adj_matrix)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


if __name__ == '__main__':
    for epoch in range(200):
        train(epoch)

    plt.subplot(1, 2, 1)
    plt.plot(range(0, 200), acc, 'o-')
    plt.plot(range(0, 200), acc_loss, '.-')

    plt.subplot(1, 2, 2)
    plt.plot(range(0, 200), vals, 'o-')
    plt.plot(range(0, 200), val_loss, '.-')
    plt.show()



    val()


