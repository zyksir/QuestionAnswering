#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
model = None

best_model_path = "./models/best.pt"
model.load_state_dict(torch.load("./models/best.pt"))

model.eval()
with torch.no_grad():
    all_score, all_label, APlist, RRlist, log = [], [], [], [], {}
    for question, question_length, positive_answer, positive_answer_length, negative_answer, negative_answer_length in valid_dataloader:
        score_label = []
        if args.cuda:
            question = question.cuda()
            positive_answer = positive_answer.cuda()
            negative_answer = negative_answer.cuda()

        if positive_answer_length is not None:
            pos_num = positive_answer.shape[0]
            question_extend = question.expand(pos_num, -1)
            question_length_extend = question_length.expand(pos_num)
            positive_score = model(question_extend, positive_answer, question_length_extend,
                                   positive_answer_length).detach().tolist()
            score_label.extend([(score, 1) for score in positive_score])

        if negative_answer_length is not None:
            neg_num = negative_answer.shape[0]
            question_extend = question.expand(neg_num, -1)
            question_length_extend = question_length.expand(neg_num)
            negative_score = model(question_extend, negative_answer, question_length_extend,
                                   negative_answer_length).detach().tolist()
            score_label.extend([(score, 0) for score in negative_score])

        score_label = sorted(score_label, reverse=True)
        rank, right_num, cur_list = 0, 0, []
        for score, label in score_label:
            all_score.append(score)
            all_label.append(label)
            rank += 1
            if label == 1:
                right_num += 1
                cur_list.append(float(right_num) / rank)
        RRlist.append(cur_list[0])
        APlist.append(float(sum(cur_list)) / len(cur_list))

MRR = float(sum(RRlist)) / len(RRlist)
log["MRR"] = MRR
MAP = float(sum(APlist)) / len(APlist)
log["MAP"] = MAP

all_score = np.array(all_score) > 0.5
all_label = np.array(all_label)
precision = precision_score(all_label, all_score)
recall = recall_score(all_label, all_score)
f1 = f1_score(all_label, all_score)