#!/usr/bin/env python
#-*- coding: utf-8 -*-

import torch
import pickle
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
        self.pos, self.q2pos_num, self.q2neg = self.get_train_data(filename)
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
            subsampling_weight
        '''
        positive_sample = self.pos[idx]
        question, positive_answer = positive_sample
        negative_answer_candidate = random.sample(self.q2neg[question], self.negative_sample_size)
        subsampling_weight = len(self.q2neg[question]) / self.q2pos_num[question]
        return question, positive_answer, negative_answer_candidate, subsampling_weight

    @staticmethod
    def collate_fn(data):
        return None

    @staticmethod
    def get_train_data(filename):
        all_triples = pickle.load(open(filename))
        q_pos = []
        q2neg = defaultdict(lambda: [])
        q2pos_num = defaultdict(lambda: 0)
        for line in all_triples:
            if line[2] == "0":
                q2neg[line[0]].append(line[1])
            else:
                q_pos.append((line[0], line[1]))
                q2pos_num[line[0]] += 1

        return q_pos, q2pos_num, q2neg
