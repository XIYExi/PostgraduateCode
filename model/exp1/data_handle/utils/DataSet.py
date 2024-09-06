import random
import copy
import torch
from torch.utils.data import Dataset
import math
import numpy as np
import random

def neg_sample(item_set, item_size):  # random sample an item id that is not in the user's interact history
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item


class CDDRecDataset(Dataset):
    def __init__(self, user_seq, max_seq_length, item_size, test_neg_items=None, data_type='train'):
        self.user_seq = user_seq  # u-i 交互矩阵
        self.test_neg_items = test_neg_items  # None
        self.data_type = data_type  # default train
        self.max_len = max_seq_length  # default 20
        self.item_size = item_size

    def __len__(self):
        return len(self.user_seq)

    def __getitem__(self, index):
        user_id = index
        items = self.user_seq[index]
        # assert self.data_type in {"train", "valid", "test"}
        input_ids = items[:-1]
        target_pos = items[1:]
        answer = [items[-1]]

        # 负样本
        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.item_size))

        # if data_argumentation ?

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        # for long sequences that longer than max_len
        input_ids = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        # test_neg_items = None
        cur_tensors = (
            torch.tensor(user_id, dtype=torch.long),     # user_id for testing
            torch.tensor(input_ids, dtype=torch.long),   # training
            torch.tensor(target_pos, dtype=torch.long),  # targeting, one item right-shifted, since task is to predict next item
            torch.tensor(target_neg, dtype=torch.long),  # random sample an item out of training and eval for every training items.
            torch.tensor(answer, dtype=torch.long),      # last item for prediction.
        )
        return cur_tensors

