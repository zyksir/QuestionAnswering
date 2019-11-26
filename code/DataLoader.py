#!/usr/bin/env python
#-*- coding: utf-8 -*-

import random
from collections import defaultdict
from torch.utils.data import Dataset

def transform_data(filename):
    '''
    :param filename:
    [自然语言问题，自然语言答案，标签]
    [[数字序列], [数字序列], 0或者1]
    :return:
    '''
    pass

class TrainDataset(Dataset):
    def __init__(self, filename, negative_sample_size, mode="WeightedLoss"):
        super(TrainDataset, self).__init__()
        self.pos, self.q2neg = self.get_train_data(filename)
        self.len = len(self.pos)
        self.mode = mode
        self.negative_sample_size = negative_sample_size
        self.seq_size = 0
        assert self.mode in ["WeightedLoss", "SelfAdversarial"]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        '''
        :param idx:
        :return:
            question: [seq_size]
            pos_answer: [seq_size]
            neg_answer: [negative_sample_size, seq_size]
        '''
        positive_sample = self.pos[idx]
        question, answer = positive_sample
        negative_answer_candidate = self.q2neg[question]
        negative_sample_list = random.sample(negative_answer_candidate, self.negative_sample_size)


    @staticmethod
    def get_train_data(filename):
        q_pos = []
        q2neg = defaultdict(lambda: [])
        with open(filename, "r") as f:
            for line in f:
                line = line.strip().split("\t")
                if line[2] == "0":
                    q2neg[line[0]].append(line[1])
                else:
                    q_pos.append((line[0], line[1]))

        return q_pos, q2neg
