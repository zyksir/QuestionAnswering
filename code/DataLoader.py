#!/usr/bin/env python
#-*- coding: utf-8 -*-

import torch
import pickle
import random
from collections import defaultdict
from torch.utils.data import Dataset

PAD_INDEX = 0
START_INDEX = 1
END_INDEX = 2
def transform_data(filename):
    '''
    :param filename:
    [自然语言问题，自然语言答案，标签]
    [[数字序列], [数字序列], 0或者1]
    :return:
    '''
    pass

class TrainDataset(Dataset):
    def __init__(self, filename, negative_sample_size, mode="UpSampling"):
        super(TrainDataset, self).__init__()
        self.pos, self.q2pos_num, self.q2neg = self.get_train_data(filename)
        self.len = len(self.pos)
        self.mode = mode
        self.negative_sample_size = negative_sample_size
        self.seq_size = 0
        assert self.mode in ["UpSampling", "SelfAdversarial"]

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
        question, positive_answer = positive_sample
        negative_answer_candidate = random.sample(self.q2neg[question], self.negative_sample_size)
        return question, positive_answer, negative_answer_candidate

    @staticmethod
    def collate_fn(data):
        que_len, pos_len, neg_len = 0, 0, 0
        batch_size = len(data)
        for q, pa, nas, _ in data:
            que_len = max(que_len, len(q))
            pos_len = max(pos_len, len(pa))
            negative_sample_size = len(nas)
            for na in nas:
                neg_len = max(neg_len, len(na))
        question = torch.LongTensor(batch_size, que_len).fill_(PAD_INDEX)
        positive_answer = torch.LongTensor(batch_size, pos_len).fill_(PAD_INDEX)
        negative_answer_candidate = torch.LongTensor(batch_size, negative_sample_size, neg_len).fill_(PAD_INDEX)
        for i, (q, pa, nas) in enumerate(data):
            question[i, 0:len(q)] = torch.LongTensor(q)
            positive_answer[i, 0:len(pa)] = torch.LongTensor(pa)
            for j, na in enumerate(nas):
                negative_answer_candidate[i, j, 0:len(na)] = torch.LongTensor(na)
        return question, positive_answer, negative_answer_candidate

    @staticmethod
    def get_train_data(filename):
        all_triples = pickle.load(open(filename))
        q_pos = []
        q2neg = defaultdict(lambda: [])
        q2pos_num = defaultdict(lambda: 0)
        for line in all_triples:
            if line[2] == 0:
                q2neg[line[0]].append(line[1])
            else:
                q_pos.append((line[0], line[1]))
                q2pos_num[line[0]] += 1

        return q_pos, q2pos_num, q2neg


class TestDataset(Dataset):
    def __init__(self, filename):
        super(TestDataset, self).__init__()
        self.triples = pickle.load(open(filename))
        self.len = len(self.triples)
        self.seq_size = 0

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        '''
        :param idx:
        :return:
            question: [seq_size]
            pos_answer: [seq_size]
            label: 1 or 0
        '''
        question, positive_answer, label = self.triples[idx]
        return question, positive_answer, label

    @staticmethod
    def collate_fn(data):
        que_len, ans_len = 0, 0
        batch_size = len(data)
        for q, a, _ in data:
            que_len = max(que_len, len(q))
            ans_len = max(ans_len, len(a))
        question = torch.LongTensor(batch_size, que_len).fill_(PAD_INDEX)
        answer = torch.LongTensor(batch_size, ans_len).fill_(PAD_INDEX)
        label = []
        for i, (q, a, l) in enumerate(data):
            question[i, 0:len(q)] = torch.LongTensor(q)
            answer[i, 0:len(a)] = torch.LongTensor(a)
            label.append(l)
        return question, answer, torch.LongTensor(label)
