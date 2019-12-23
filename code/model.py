#!/usr/bin/env python
#-*- coding: utf-8 -*-

from IPython import embed
import random
import numpy as np
import torch
import logging
from utils import LSTM
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, precision_recall_curve
from UtilsDCN import get_pretrained_embedding, init_lstm_forget_bias, DCNEncoder, DCNFusionBiLSTM 

def inverse_sigmoid(score):
    return -torch.log(1/(score+1e-7) - 1)

class MarginLoss(nn.Module):
    def __init__(self, adv_temperature=None, margin=0.75):
        super(MarginLoss, self).__init__()
        self.margin = nn.Parameter(torch.Tensor([margin]))
        self.margin.requires_grad = False
        if adv_temperature != None:
            self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
            self.adv_temperature.requires_grad = False
            self.adv_flag = True
        else:
            self.adv_flag = False

    def get_weights(self, n_score):
        return F.softmax(n_score * self.adv_temperature, dim=-1).detach()

    def forward(self, p_score, n_score):
        if self.adv_flag:
            return (self.get_weights(n_score) * torch.max(n_score - p_score, -self.margin)).sum(
                dim=-1).mean() + self.margin
        else:
            return (torch.max(n_score - p_score, -self.margin)).mean() + self.margin

    def predict(self, p_score, n_score):
        score = self.forward(p_score, n_score)
        return score.cpu().data.numpy()

class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.config = args
        self.embedding = nn.Embedding(args.word_num, args.embed_dim)
        # class_weights = torch.FloatTensor(eval(args.class_weights))
        # if args.cuda:
        #     class_weights = class_weights.cuda()
        # self.loss_func = nn.CrossEntropyLoss(weight=class_weights)
        self.loss_func = MarginLoss(args.adv_temperature, margin=args.margin)

    def forward(self, question, answer, question_length, answer_length):
        '''
        :param
            question: (batch_size, que_size)
            answer: (batch_size, ans_size)
            question_length: (batch_size)
            answer_length: (batch_size)
        :return:
            score: (batch_size, 2)
        '''
        raise NotImplementedError

    @staticmethod
    def do_train(model, optimizer, train_dataloader, args, lr_scheduler=None, warmup_scheduler=None):
        '''
        :param
            pos_question: (batch_size, que_size)
            pos_answer: (batch_size, pos_size)
            neg_question: (batch_size*negative_sample_size, que_size2)
            neg_answer: (batch_size*negative_sample_size, neg_size)
        '''
        model.train()
        count = 0
        log_loss = 0
        for positive_question, positive_question_length, positive_answer, positive_answer_length, \
               negative_question, negative_question_length, negative_answer, negative_answer_length, if_pos in train_dataloader:
            count += 1
            optimizer.zero_grad()
            batch_size = positive_answer.shape[0]
            negative_sample_size = negative_answer.shape[0] // batch_size
            if args.cuda:
                positive_question = positive_question.cuda()
                positive_answer = positive_answer.cuda()
                negative_question = negative_question.cuda()
                negative_answer = negative_answer.cuda()
            positive_score = model.forward(positive_question, positive_answer, positive_question_length, positive_answer_length)
            negative_score = model.forward(negative_question, negative_answer, negative_question_length, negative_answer_length)

            positive_score, negative_score = positive_score.view(-1), negative_score.view(-1)
            positive_score = positive_score.repeat(negative_sample_size)
            loss = model.loss_func(positive_score, negative_score) if if_pos else model.loss_func(negative_score, positive_score)

            # target = torch.cat([torch.ones(positive_score.size()), torch.zeros(negative_score.size())])
            # if args.cuda:
            #     target = target.cuda()
            # loss = F.binary_cross_entropy(torch.cat([positive_score, negative_score]), target)

            # target = torch.cat([torch.ones(positive_score.size()), torch.zeros(negative_score.size())])
            # if args.cuda:
            #     target = target.cuda()
            # loss = model.loss_func(torch.cat([positive_score, negative_score]), target)

            loss.backward()
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()
            if warmup_scheduler:
                warmup_scheduler.dampen()
            log_loss += loss.item()
            if count % model.config.log_step == 0:
                logging.info("count: %d, loss: %s" % (count, log_loss/model.config.log_step))
                log_loss = 0
                # if random.uniform(0, 1) < 0.01:
                # print("######################################################################")
                # print(positive_score)

    @staticmethod
    def do_valid(model, valid_dataloader, args):
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
                    positive_score = model(question_extend, positive_answer, question_length_extend, positive_answer_length).detach().tolist()
                    score_label.extend([(score, 1) for score in positive_score])

                if negative_answer_length is not None:
                    neg_num = negative_answer.shape[0]
                    question_extend = question.expand(neg_num, -1)
                    question_length_extend = question_length.expand(neg_num)
                    negative_score = model(question_extend, negative_answer, question_length_extend, negative_answer_length).detach().tolist()
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

        all_score = np.array(all_score)
        all_label = np.array(all_label)
        log["MRR"] = float(sum(RRlist)) / len(RRlist)
        log["MAP"] = float(sum(APlist)) / len(APlist)
        log["Average Positive Score"] = all_score[all_label == 1].mean()
        log["Average Negative Score"] = all_score[all_label == 0].mean()

        threshold = (log["Average Positive Score"]*args.negative_sample_size+log["Average Negative Score"]) / (args.negative_sample_size+1)
        log["precision"] = precision_score(all_label, all_score > threshold)
        log["recall"] = recall_score(all_label, all_score > threshold)
        log["f1"] = f1_score(all_label, all_score > threshold)
        log["threshold"] = threshold
        #
        # precision, recall, threshold = precision_recall_curve(all_label, all_score)
        # f1_scores = 2 * recall * precision / (recall + precision)
        # index = np.argmax(f1_scores)
        # log["precision"] = precision[index]
        # log["recall"] = recall[index]
        # log["f1"] = f1_scores[index]
        # log["threshold"] = threshold[index]

        return log

    @staticmethod
    def do_prediction(model, valid_dataloader, args, word2id):
        # model.load_state_dict(torch.load(args.best_model_path))
        log = model.do_valid(model, valid_dataloader, args)
        threshold = log["threshold"]
        sentence2question_id, sentence2answer_id = {}, {}
        with open(args.test_file, "r") as f, open(args.predict_file, "w") as fw:
            import jieba
            jieba.load_userdict("./data/pretrained_word.txt")
            for line in tqdm(f):
                line = line.strip().split("\t")
                if line[0] in sentence2question_id:
                    question = sentence2question_id[line[0]]
                else:
                    question_sentence = jieba.cut(line[0], cut_all=False)
                    question = [word2id["[STARTq]"]]
                    for word in question_sentence:
                        question.append(word2id[word] if word in word2id else word2id["[UNKNOWNq]"])
                    question.append(word2id["[ENDq]"])

                if line[1] in sentence2answer_id:
                    answer = sentence2answer_id[line[1]]
                else:
                    answer_sentence = jieba.cut(line[1], cut_all=False)
                    answer = [word2id["[STARTa]"]]
                    for word in answer_sentence:
                        answer.append(word2id[word] if word in word2id else word2id["[UNKNOWNa]"])
                    answer.append(word2id["[ENDa]"])

                question_length, answer_length = torch.LongTensor([len(question)]), torch.LongTensor([len(answer)])
                question, answer = torch.LongTensor(question).unsqueeze(0), torch.LongTensor(answer).unsqueeze(0)
                if args.cuda:
                    question, answer = question.cuda(), answer.cuda()
                score = model(question, answer, question_length, answer_length)
                fw.write("%s\t%s\t%f\t%d\n" % (line[0], line[1], score.item(), int(score>threshold)))
        return log


class CNNModel(BaseModel):
    def __init__(self, args):
        super(CNNModel, self).__init__(args)
        self.conv = nn.Sequential(
            nn.Conv2d(1, self.config.channel_size, (self.config.conv_kernel_1, self.config.conv_kernel_2), stride=1,
                      padding=(self.config.conv_kernel_1 // 2, self.config.conv_kernel_2 // 2)),
            # channel_in=1, channel_out=8, kernel_size=3*3
            nn.ReLU(True))

        self.pooling1 = nn.MaxPool2d((self.config.que_max_len, 1),
                                    stride=(self.config.que_max_len, 1), padding=0)

        self.pooling2 = nn.MaxPool2d((1, self.config.ans_max_len),
                                     stride=(1, self.config.ans_max_len), padding=0)
        self.fc1 = nn.Sequential(
            nn.Linear(self.config.que_max_len * self.config.channel_size, 300),
            nn.ReLU(),
            nn.Dropout(p=self.config.dropout),
            nn.Linear(300, 1))

        self.fc2 = nn.Sequential(
            nn.Linear(self.config.ans_max_len * self.config.channel_size, 300),
            nn.ReLU(),
            nn.Dropout(p=self.config.dropout),
            nn.Linear(300, 1))

        self.fc = nn.Linear(2, 1)

    def dynamic_pooling_index(self, len1, len2, max_len1, max_len2):
        def dpool_index_(batch_idx, len1_one, len2_one, max_len1, max_len2):
            stride1 = 1.0 * max_len1 / len1_one
            stride2 = 1.0 * max_len2 / len2_one
            idx1_one = [int(i/stride1) for i in range(max_len1)]
            idx2_one = [int(i/stride2) for i in range(max_len2)]
            return idx1_one, idx2_one
        batch_size = len1.shape[0]
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

    def matchPyramid(self, question, answer, question_length, answer_length):
        '''
        param:
            question: (batch_size, que_len, embed_dim)
            answer: (batch_size, ans_len, embed_dim)
        return:
            score: (batch, 1)
        '''
        batch_size, que_len, embed_dim = question.shape
        ans_len = answer.shape[1]
        if que_len > self.config.que_max_len:
            question = question[:, 0:self.config.que_max_len, :]
        if ans_len > self.config.ans_max_len:
            answer = answer[:, 0:self.config.ans_max_len, :]

        answer_trans = torch.transpose(answer, 1, 2) # (batch_size, embed_dim, ans_len)
        question_norm = torch.sqrt(torch.sum(question * question, dim=2, keepdim=True))
        answer_norm = torch.sqrt(torch.sum(answer_trans*answer_trans, dim=1, keepdim=True))
        cross = torch.bmm(question/question_norm, answer_trans/answer_norm).unsqueeze(1) # (batch, 1, seq_len, rel_len)

        # (batch, channel_size, seq_len, rel_len)
        conv1 = self.conv(cross)
        channel_size = conv1.size(1)
        conv1_expand = conv1.repeat(1, 1, (self.config.que_max_len // que_len) + 1, (self.config.ans_max_len // ans_len) + 1)[:, :, :self.config.que_max_len, :self.config.ans_max_len]

        pool1 = self.pooling1(conv1_expand).view(batch_size, -1) # (batch, channel_size, p_size1, p_size2)
        out1 = self.fc1(pool1) # (batch, 1)

        pool2 = self.pooling2(conv1_expand).view(batch_size, -1)
        out2 = self.fc2(pool2)

        return out1, out2

    def forward(self, question, answer, question_length, answer_length):
        '''
        :param
            question: (batch_size, que_size)
            answer: (batch_size, ans_size)
        :return:
            score: (batch_size)
        '''
        question = self.embedding(question)
        answer = self.embedding(answer)
        score1, score2 = self.matchPyramid(question, answer, question_length, answer_length)
        score = self.fc(torch.cat((score1, score2), 1))
        return torch.sigmoid(score).view(-1)

class RNNModel(BaseModel):
    def __init__(self, args):
        super(RNNModel, self).__init__(args)
        self.question_encoder = nn.GRU(
            input_size=args.embed_dim, hidden_size=args.hidden_dim,
            num_layers=args.num_layers, dropout=args.dropout,
            bidirectional=args.birnn
        )
        self.answer_encoder = nn.GRU(
            input_size=args.embed_dim, hidden_size=args.hidden_dim,
            num_layers=args.num_layers, dropout=args.dropout,
            bidirectional=args.birnn
        )

    def encode(self, encoder, input, input_length, hidden=None):
        input = torch.transpose(input, 0, 1)
        input_length_sorted = sorted(input_length, reverse=True)
        sort_index = np.argsort(-np.array(input_length)).tolist()
        input_sorted = Variable(torch.zeros(input.size())).cuda()
        batch_size = input.size()[1]
        for b in range(batch_size):
            input_sorted[:, b, :] = input[:, sort_index[b], :]
        packed = torch.nn.utils.rnn.pack_padded_sequence(input_sorted, input_length_sorted)
        outputs, hidden = encoder(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs_resorted = Variable(torch.zeros(outputs.size())).cuda()
        hidden_resorted = Variable(torch.zeros(hidden.size())).cuda()
        for b in range(batch_size):
            outputs_resorted[:, sort_index[b], :] = outputs[:, b, :]
            hidden_resorted[:, sort_index[b], :] = hidden[:, b, :]

        hidden_resorted = torch.sum(hidden_resorted, dim=0)
        outputs_resorted = torch.transpose(outputs_resorted, 0, 1)
        return outputs_resorted, hidden_resorted

    def forward(self, question, answer, question_length, answer_length):
        question = self.embedding(question)
        question, question_hidden = self.encode(self.question_encoder, question, question_length)
        answer = self.embedding(answer)
        answer, answer_hidden = self.encode(self.answer_encoder, answer, answer_length)

        question = torch.tanh(self.max_pooling3D(question))
        answer = torch.tanh(self.max_pooling3D(answer))
        score = torch.cosine_similarity(question, answer, dim=1)
        # score = torch.sum(question * answer, 1)

        return score

    @staticmethod
    def max_pooling3D(sentence):
        '''
        :param lstm_out: (batch_size, seq_len, hidden_dim)
        :return: (batch_size, hidden_dim)
        '''
        batch_size, seq_len, hidden_dim = sentence.shape
        sentence = sentence.unsqueeze(-1)
        maxpooling = nn.MaxPool3d(kernel_size=[seq_len, 1, 1], stride=[1, 1, 1], padding=0)
        output = maxpooling(sentence)
        output = output.view(batch_size, hidden_dim)
        return output

class CNNRNNModel(CNNModel):
    def __init__(self, args):
        super(CNNRNNModel, self).__init__(args)
        self.question_encoder = nn.GRU(
            input_size=args.embed_dim, hidden_size=args.hidden_dim,
            num_layers=args.num_layers, dropout=args.dropout,
            bidirectional=args.birnn
        )
        self.answer_encoder = nn.GRU(
            input_size=args.embed_dim, hidden_size=args.hidden_dim,
            num_layers=args.num_layers, dropout=args.dropout,
            bidirectional=args.birnn
        )
        self.fc = nn.Linear(3, 1)

    def encode(self, encoder, input, input_length, hidden=None):
        input = torch.transpose(input, 0, 1)
        input_length_sorted = sorted(input_length, reverse=True)
        sort_index = np.argsort(-np.array(input_length)).tolist()
        input_sorted = Variable(torch.zeros(input.size()))
        if self.config.cuda:
            input_sorted = input_sorted.cuda()
        batch_size = input.size()[1]
        for b in range(batch_size):
            input_sorted[:, b, :] = input[:, sort_index[b], :]
        packed = torch.nn.utils.rnn.pack_padded_sequence(input_sorted, input_length_sorted)
        outputs, hidden = encoder(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs_resorted = Variable(torch.zeros(outputs.size()))
        hidden_resorted = Variable(torch.zeros(hidden.size()))
        if self.config.cuda:
            outputs_resorted = outputs_resorted.cuda()
            hidden_resorted = hidden_resorted.cuda()
        for b in range(batch_size):
            outputs_resorted[:, sort_index[b], :] = outputs[:, b, :]
            hidden_resorted[:, sort_index[b], :] = hidden[:, b, :]

        hidden_resorted = torch.sum(hidden_resorted, dim=0)
        outputs_resorted = torch.transpose(outputs_resorted, 0, 1)
        return outputs_resorted, hidden_resorted

    def forward(self, question, answer, question_length, answer_length):
        question = self.embedding(question)
        answer = self.embedding(answer)
        score1, score2 = self.matchPyramid(question, answer, question_length, answer_length)

        question, question_hidden = self.encode(self.question_encoder, question, question_length)
        answer, answer_hidden = self.encode(self.answer_encoder, answer, answer_length)
        question = torch.tanh(self.max_pooling3D(question))
        answer = torch.tanh(self.max_pooling3D(answer))
        score3 = torch.sum(question * answer, 1, keepdim=True)
        score = self.fc(torch.cat((score1, score2, score3), 1))
        # if random.uniform(0, 1)<0.0001:
        #     print(score1.mean())
        #     print(score2.mean())
        #     print(score3.mean())
        #     print(score.mean())
        return torch.sigmoid(score.view(-1))

    @staticmethod
    def max_pooling3D(sentence):
        '''
        :param lstm_out: (batch_size, seq_len, hidden_dim)
        :return:
        '''
        batch_size, seq_len, hidden_dim = sentence.shape
        sentence = sentence.unsqueeze(-1)
        maxpooling = nn.MaxPool3d(kernel_size=[seq_len, 1, 1], stride=[1, 1, 1], padding=0)
        output = maxpooling(sentence)
        output = output.view(batch_size, hidden_dim)
        return output

class CoattentionModel(BaseModel):
    def __init__(self, args):
        super(CoattentionModel, self).__init__(args)
        '''
        params:
            hidden_dim: the hidden size of LSTM networks
            emb_dim: the embedding size of words 
            dropout_ratio: the dropout rate for encoder layer and fusion layer
        '''
        self.hidden_dim = self.config.hidden_dim
        self.emb_dim = self.config.embed_dim
        self.dropout_ratio = self.config.dropout
        self.if_bidirec_init = 0

        self.encoder = DCNEncoder(self.hidden_dim, self.embedding, self.dropout_ratio, self.if_bidirec_init)
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fusion_lstm = DCNFusionBiLSTM(self.hidden_dim, self.dropout_ratio)
        self.output = nn.Linear(2 * self.hidden_dim, 1)

        self.dropout = nn.Dropout(p=self.dropout_ratio)

    def max_pooling3D(self, sentence):
        '''
        :param lstm_out: (batch_size, seq_len, hidden_dim)
        :return: (batch_size, hidden_dim)
        '''
        batch_size, seq_len, hidden_dim = sentence.shape
        sentence = sentence.unsqueeze(-1)
        maxpooling = nn.MaxPool3d(kernel_size=[seq_len, 1, 1], stride=[1, 1, 1], padding=0)
        output = maxpooling(sentence)
        output = output.view(batch_size, hidden_dim)
        return output

    def forward(self, q_seq, d_seq, q_mask, d_mask):
        '''
        input alias:
            q_seq: question words
            d_seq: answer words
            q_mask: boolen question length (batch_size, batch_question_seq_size)
            d_mask: boolen answer lengths (batch_size, batch_answer_seq_size)
        '''
        # 1. Encode question and document
        # size: (batch_size, question_size + 1, embedding_size)
        Q = self.encoder(q_seq, q_mask)
        # size: (batch_size, document_size + 1, embedding_size)
        D = self.encoder(d_seq, d_mask)

        # 2. Project Q
        # Allow for variation between question encoding space and document encoding space
        # Linear layer + Activation layer to project Q
        # size: (batch_size, question_size + 1, embedding_size)
        Q = torch.tanh(self.q_proj(Q.view(-1, self.hidden_dim))).view(Q.size())

        # 3. Calculate attention
        # Calculate similarity matrix L
        # size: (batch_size, embedding_size, document_size+1)
        D_t = torch.transpose(D, 1, 2)
        # size: (batch_size, question_size+1, document_size+1)
        L = torch.bmm(Q, D_t)

        # A_Q: for each word in the question, m+1 document word attention add up to 1
        A_Q_ = F.softmax(L, dim=1)  # (B, n+1, m+1)
        A_Q = torch.transpose(A_Q_, 1, 2)  # (B, m+1, n+1)
        C_Q = torch.bmm(D_t, A_Q)  # (B, l, n+1)

        # A_D: for each word in the document, n+1 question word attention add up to 1
        Q_t = torch.transpose(Q, 1, 2)  # (B, l, n+1)
        A_D = F.softmax(L, dim=2)  # (B, n+1, m+1)
        C_D = torch.bmm(torch.cat((Q_t, C_Q), 1), A_D)  # (B, 2l, m+1)

        C_D_t = torch.transpose(C_D, 1, 2)  # (B, m+1, 2l)

        # 4. Fusion BiLSTM
        # size: (batch_size, document_size+1, 3l)
        # mark: kind of transpose from original figure
        bilstm_in = torch.cat((C_D_t, D), 2)
        bilstm_in = self.dropout(bilstm_in)
        # final embedding
        # size: (batch_size, seq_len, 2*embedding_size)
        U = self.fusion_lstm(bilstm_in, d_mask)

        final_embedding = self.max_pooling3D(U)

        out = torch.sigmoid(self.output(final_embedding))

        return out

class BIDAF(BaseModel):
    def __init__(self,args):
        super(BIDAF, self).__init__(args)
        '''
        already declared in BaseModel:
        self.config = args
        self.embedding = nn.Embedding(args.word_num, args.embed_dim)
        self.loss_func = MarginLoss(args.adv_temperature, margin=args.margin)
        '''
        self.context_LSTM = LSTM(
            input_size = args.embed_dim,
            hidden_size=args.hidden_dim,
            num_layers=1,
            dropout=args.dropout
        )
        self.att_Linear = nn.Linear(args.hidden_dim, 1)
        self.att_Linear2 = nn.Linear(args.hidden_dim,1)
        self.att_Linear3 = nn.Linear(args.hidden_dim,1)
        self.model_LSTM = LSTM(
            input_size=4 * args.hidden_dim,
            hidden_size=args.hidden_dim,
            num_layers=2,
            dropout=args.dropout,
            batch_first=True,
            bidirectional=True
        )
        self.output_Linear = nn.Linear(6 * args.hidden_dim,1)
        return

    def cal_para(self):
        size = 0
        for para in self.parameters():
            tmp = 1
            for i in para.size():
                tmp *=i
            size+=tmp
        print("para num = ",size)

    # def weight_init(self):
    #     for module in self.modules():
    #         if(isinstance(module,nn.Linear)):
    #             nn.init.constant_(module.bias)
    #             nn.init.uniform_(module.weight,-0.1,0.1)
    #         elif(isinstance(module,LSTM)):
    #             nn.init.uniform_(module.lstm.all_weights,-0.1,0.1)
    #     return

    def max_pooling3D(self,input):
        batch_size,seq_len,hidden_dim=input.shape
        input = input.unsqueeze(-1)
        maxpooling = nn.MaxPool3d(
            kernel_size=[seq_len,1,1],
            stride=[1,1,1],
            padding=0
        )
        output = maxpooling(input)
        output = output.view(batch_size,hidden_dim)
        return output

    def attention_flow(self,context,question):
        lenc = context.shape[1]
        lenq = question.shape[1]

        S = []
        for i in range(lenq):
            qi = question[:,i:i+1,:]
            #[batch_size,lenc,hidden_dim]
            cqi = qi * context
            cqi = self.att_Linear(cqi)
            S.append(cqi)
        S = torch.cat(S,dim=-1)
        S += self.att_Linear2(context) + (self.att_Linear3(question).transpose(1,2))
        a = F.softmax(S,dim=-1)
        c2q = torch.bmm(a,question)
        b = F.softmax(torch.max(S,dim=2)[0], dim=-1).unsqueeze(1)
        q2c = torch.bmm(b,context).expand(-1,lenc,-1)
        output = torch.cat([context,c2q,context * q2c,context * c2q],dim=-1)
        return output


    def forward(self,question,context,question_length,context_length):
        # self.cal_para()
        question = self.embedding(question)
        context=self.embedding(context)

        question = self.context_LSTM(question,question_length)
        context = self.context_LSTM(context,context_length)
        # print(context)

        G = self.attention_flow(context,question)

        M = self.model_LSTM(G,context_length)

        final_embed = torch.cat([G,M],-1)
        # final_embed = self.max_pooling3D(G)
        final_embed = final_embed.mean(dim=1)

        output = self.output_Linear(final_embed)

        output = torch.sigmoid(output)

        return output
