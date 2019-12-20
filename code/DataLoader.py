#!/usr/bin/env python
#-*- coding: utf-8 -*-

import torch
import pickle
import random
from collections import defaultdict
from torch.utils.data import Dataset
from IPython import embed

PAD_A = 4
PAD_Q = 0

class TrainDataset(Dataset):
    def __init__(self, filename, negative_sample_size, word2id, if_pos=True):
        super(TrainDataset, self).__init__()
        self.pos, self.neg = self.get_train_data(filename, if_pos=if_pos)
        self.len = len(self.pos)
        self.if_pos = if_pos
        self.negative_sample_size = negative_sample_size
        self.word2id = word2id

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
        negative_candidate = random.sample(self.neg, self.negative_sample_size)
        return positive_sample, negative_candidate, self.if_pos

    @staticmethod
    def collate_fn(data):
        batch_size = len(data)
        negative_sample_size = len(data[0][1])
        max_pos_que_len, max_pos_ans_len, max_neg_que_len, max_neg_ans_len = 0, 0, 0, 0
        for pos_sample, neg_candidate, if_pos in data:
            max_pos_que_len = max(max_pos_que_len, len(pos_sample[0]))
            max_pos_ans_len = max(max_pos_ans_len, len(pos_sample[1]))
            for neg_que, neg_ans in neg_candidate:
                max_neg_que_len = max(max_neg_que_len, len(neg_que))
                max_neg_ans_len = max(max_neg_ans_len, len(neg_ans))
        positive_question = torch.LongTensor(batch_size, max_pos_que_len).fill_(PAD_Q)
        positive_answer = torch.LongTensor(batch_size, max_pos_ans_len).fill_(PAD_A)
        positive_question_length = torch.LongTensor(batch_size).fill_(1)
        positive_answer_length = torch.LongTensor(batch_size).fill_(1)

        negative_question = torch.LongTensor(batch_size * negative_sample_size, max_neg_que_len).fill_(PAD_Q)
        negative_answer = torch.LongTensor(batch_size * negative_sample_size, max_neg_ans_len).fill_(PAD_A)
        negative_question_length = torch.LongTensor(batch_size * negative_sample_size).fill_(1)
        negative_answer_length = torch.LongTensor(batch_size * negative_sample_size).fill_(1)
        for i, (pos_sample, neg_candidate, _) in enumerate(data):
            index = i
            positive_question[index, 0:len(pos_sample[0])] = torch.LongTensor(pos_sample[0])
            positive_question_length[index] = len(pos_sample[0])
            positive_answer[index, 0:len(pos_sample[1])] = torch.LongTensor(pos_sample[1])
            positive_answer_length[index] = len(pos_sample[1])
            for j, (neg_que, neg_ans) in enumerate(neg_candidate):
                index = i*negative_sample_size + j
                negative_question[index, 0:len(neg_que)] = torch.LongTensor(neg_que)
                negative_question_length[index] = len(neg_que)
                negative_answer[index, 0:len(neg_ans)] = torch.LongTensor(neg_ans)
                negative_answer_length[index] = len(neg_ans)
        return positive_question, positive_question_length, positive_answer, positive_answer_length, \
               negative_question, negative_question_length, negative_answer, negative_answer_length, if_pos

    @staticmethod
    def get_train_data(filename, if_pos):
        all_triples = pickle.load(open(filename, "rb"))
        pos, neg = [], []
        for line in all_triples:
            if line[2] == if_pos:
                pos.append((line[0], line[1]))
            else:
                neg.append((line[0], line[1]))
        return pos, neg

# class TestDataset(Dataset):
#     def __init__(self, filename):
#         super(TestDataset, self).__init__()
#         self.triples = pickle.load(open(filename, "rb"))
#         self.len = len(self.triples)
#         self.seq_size = 0
#
#     def __len__(self):
#         return self.len
#
#     def __getitem__(self, idx):
#         '''
#         :param idx:
#         :return:
#             question: [seq_size]
#             pos_answer: [seq_size]
#             label: 1 or 0
#         '''
#         question, positive_answer, label = self.triples[idx]
#         return question, positive_answer, label
#
#     @staticmethod
#     def collate_fn(data):
#         que_len, ans_len = 0, 0
#         batch_size = len(data)
#         for q, a, _ in data:
#             que_len = max(que_len, len(q))
#             ans_len = max(ans_len, len(a))
#         question = torch.LongTensor(batch_size, que_len).fill_(PAD_INDEX)
#         answer = torch.LongTensor(batch_size, ans_len).fill_(PAD_INDEX)
#         question_length = torch.LongTensor(batch_size).fill_(1)
#         answer_length = torch.LongTensor(batch_size).fill_(1)
#         label = []
#         for i, (q, a, l) in enumerate(data):
#             question[i, 0:len(q)] = torch.LongTensor(q)
#             answer[i, 0:len(a)] = torch.LongTensor(a)
#             question_length[i] = len(q)
#             answer_length[i] = len(a)
#             label.append(l)
#         return question, question_length, answer, answer_length, torch.LongTensor(label)

class TestDataset(Dataset):
    def __init__(self, filename):
        super(TestDataset, self).__init__()
        self.que, self.q2pos, self.q2neg = self.get_train_data(filename)
        self.len = len(self.que)
        self.seq_size = 0

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        '''
        :param idx:
        :return:
            que: [seq_size]
            pos_ans: [num_pos, seq_size]
            neg_ans: [num_neg, seq_size]
        '''
        que, pos_ans, neg_ans = self.que[idx], self.q2pos[idx], self.q2neg[idx]
        que_len, pos_ans_len, neg_ans_len = len(que), [], []
        question = torch.LongTensor(que).unsqueeze(0)
        question_length = torch.LongTensor(1).fill_(que_len)

        if len(pos_ans) == 0:
            positive_answer, positive_answer_length = question, None
        else:
            for a in pos_ans:
                pos_ans_len.append(len(a))
            max_pos_ans_len = max(pos_ans_len)
            positive_answer_length = torch.LongTensor(pos_ans_len)
            positive_answer = torch.LongTensor(len(pos_ans), max_pos_ans_len).fill_(PAD_A)
            for i, a in enumerate(pos_ans):
                positive_answer[i, 0:len(a)] = torch.LongTensor(a)

        if len(neg_ans) == 0:
            negative_answer, negative_answer_length = question, None
        else:
            for a in neg_ans:
                neg_ans_len.append(len(a))
            max_neg_ans_len = max(neg_ans_len)
            negative_answer_length = torch.LongTensor(neg_ans_len)
            negative_answer = torch.LongTensor(len(neg_ans), max_neg_ans_len).fill_(PAD_A)
            for i, a in enumerate(neg_ans):
                negative_answer[i, 0:len(a)] = torch.LongTensor(a)

        return question, question_length, positive_answer, positive_answer_length, negative_answer, negative_answer_length

    @staticmethod
    def collate_fn(data):
        return data[0]

    @staticmethod
    def get_train_data(filename):
        all_triples = pickle.load(open(filename, "rb"))
        que, q2pos, q2neg = [], defaultdict(lambda :[]), defaultdict(lambda :[])
        q2index = {}
        for line in all_triples:
            if str(line[0]) not in q2index:
                q2index[str(line[0])] = len(que)
                que.append(line[0])
            index = q2index[str(line[0])]
            if line[2] == 0:
                q2neg[index].append(line[1])
            else:
                q2pos[index].append(line[1])
        return que, q2pos, q2neg
