#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import  recall_score, precision_score, accuracy_score, f1_score

def inverse_sigmoid(score):
    return -torch.log(1/(score+1e-7) - 1)

class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.config = args
        self.embedding = nn.Embedding(args.word_num, args.embed_dim)

    def forward(self, *input):
        '''
        :param
            question: (batch_size, que_size)
            answer: (batch_size, ans_size)
        :return:
            score: (batch_size)
        '''
        raise NotImplementedError

    @staticmethod
    def do_train(model, optimizer, train_dataloader, args):
        '''
        :param
            question: (batch_size, que_size)
            pos_answer: (batch_size, pos_size)
            neg_answer: (batch_size, negative_sample_size, neg_size)
        '''
        model.train()
        for question, positive_answer, negative_answer_candidate in train_dataloader:
            que_size, pos_size = question.shape[1], positive_answer.shape[1]
            batch_size, negative_sample_size, neg_size = negative_answer_candidate.shape
            optimizer.zero_grad()
            if args.cuda:
                question = question.cuda()
                positive_answer = positive_answer.cuda()
                negative_answer_candidate = negative_answer_candidate.cuda()
            question_expand = question.unsqueeze(1).expand(que_size, negative_sample_size, que_size)
            positive_score = model.forward(question, positive_answer)
            negative_score = model.forward(question_expand.view(-1, que_size), negative_answer_candidate.view(-1, neg_size))
            if args.mode == "UpSampling":
                positive_score = positive_score.expand(batch_size * negative_sample_size)
            elif args.mode == "SelfAdversarial":
                negative_score = negative_score.view(batch_size, negative_sample_size)
                negative_score = (F.softmax(inverse_sigmoid(negative_score), dim=1).detach() * negative_score).sum(dim=1)
            else:
                raise ValueError("mode %s not supported" % args.mode)

            target = torch.cat([torch.ones(positive_score.size()), torch.zeros(negative_score.size())])
            if args.cuda:
                target = target.cuda()
            loss = F.binary_cross_entropy(torch.cat([positive_score, negative_score]), target)
            loss.backward()
            optimizer.step()

    @staticmethod
    def do_vaild(model, vaild_dataloader, args):
        model.eval()
        with torch.no_grad():
            all_score = []
            all_label = []
            for question, answer, label in vaild_dataloader:
                if args.cuda:
                    question = question.cuda()
                    answer = answer.cuda()
                score = model.forward(question, answer)
                all_score.extend(score.tolist())
                all_label.extend(label.tolist())
        precision = precision_score(all_label, all_score)
        recall = recall_score(all_label, all_score)
        f1 = f1_score(all_label, all_score)
        log = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        return log

class CNNModel(BaseModel):
    def __init__(self, args):
        super(CNNModel, self).__init__(args)
        self.pooling1 = nn.MaxPool2d((self.config.que_maxlen, 1),
                                    stride=(self.config.que_maxlen, 1), padding=0)

        self.pooling2 = nn.MaxPool2d((1, self.config.ans_maxlen),
                                     stride=(1, self.config.ans_maxlen), padding=0)
        self.fc1 = nn.Sequential(
            nn.Linear(self.config.que_maxlen * self.config.channel_size, 20),
            nn.ReLU(),
            nn.Dropout(p=self.config.dropout),
            nn.Linear(20, 1))

        self.fc2 = nn.Sequential(
            nn.Linear(self.config.ans_maxlen * self.config.channel_size, 20),
            nn.ReLU(),
            nn.Dropout(p=self.config.dropout),
            nn.Linear(20, 1))

        self.fc = self.fc2 = nn.Linear(2, 1)

    def dynamic_pooling_index(self, len1, len2, max_len1, max_len2):
        def dpool_index_(batch_idx, len1_one, len2_one, max_len1, max_len2):
            stride1 = 1.0 * max_len1 / len1_one
            stride2 = 1.0 * max_len2 / len2_one
            idx1_one = [int(i/stride1) for i in range(max_len1)]
            idx2_one = [int(i/stride2) for i in range(max_len2)]
            return idx1_one, idx2_one
        batch_size = len(len1)
        index1, index2 = [], []
        for i in range(batch_size):
            idx1_one, idx2_one = dpool_index_(i, len1[i], len2[i], max_len1, max_len2)
            index1.append(idx1_one)
            index2.append(idx2_one)
        index1 = torch.LongTensor(index1)
        index2 = torch.LongTensor(index2)
        if self.config.cuda:
            index1 = index1.cuda()
            index2 = index2.cuda()
        return Variable(index1), Variable(index2)

    def matchPyramid(self, question, answer):
        '''
        param:
            question: (batch_size, que_len, embed_dim)
            answer: (batch_size, ans_len, embed_dim)
        return:
            score: (batch, 1)
        '''
        batch_size, que_len, embed_dim = question.shape
        ans_len = answer.shape[1]
        if que_len > self.config.que_maxlen:
            question = question[:, 0:self.config.que_maxlen, :]
            que_len = self.config.que_maxlen
        if ans_len > self.config.ans_maxlen:
            answer = answer[:, 0:self.config.ans_maxlen, :]
            ans_len = self.config.ans_maxlen

        answer_trans = torch.transpose(answer, 1, 2) # (batch_size, embed_dim, ans_len)
        question_norm = torch.sqrt(torch.sum(question * question, dim=2, keepdim=True))
        answer_norm = torch.sqrt(torch.sum(answer_trans*answer_trans, dim=1, keepdim=True))
        cross = torch.bmm(question/question_norm, answer_trans/answer_norm).unsqueeze(1) # (batch, 1, seq_len, rel_len)

        # (batch, channel_size, seq_len, rel_len)
        conv1 = self.conv(cross)
        channel_size = conv1.size(1)

        # dpool_index1: (batch, que_maxlen)
        # dpool_index2: (batch, ans_maxlen)
        dpool_index1, dpool_index2 = self.dynamic_pooling_index(que_len, ans_len, self.seq_maxlen, self.rel_maxlen)
        dpool_index1 = dpool_index1.unsqueeze(1).unsqueeze(-1).expand(batch_size, channel_size,
                                                                self.seq_maxlen, self.rel_maxlen)
        dpool_index2 = dpool_index2.unsqueeze(1).unsqueeze(2).expand_as(dpool_index1)
        conv1_expand = torch.gather(conv1, 2, dpool_index1)
        conv1_expand = torch.gather(conv1_expand, 3, dpool_index2)

        pool1 = self.pooling1(conv1_expand).view(batch_size, -1) # (batch, channel_size, p_size1, p_size2)
        out1 = self.fc1(pool1) # (batch, 1)

        pool2 = self.pooling2(conv1_expand).view(batch_size, -1)
        out2 = self.fc2(pool2)

        return out1, out2

    def forward(self, question, answer):
        '''
        :param
            question: (batch_size, que_size)
            answer: (batch_size, ans_size)
        :return:
            score: (batch_size)
        '''
        question = self.embedding(question)
        answer = self.embedding(question)
        score1, score2 = self.matchPyramid(question, answer)
        score = torch.cat((score1, score2), 2)
        score = self.fc(score)
        return score
