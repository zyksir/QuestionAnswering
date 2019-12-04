#!/usr/bin/env python
#-*- coding: utf-8 -*-
'''
输入：一个文件，三元组形式：[自然语言问题，自然语言答案，标签]
输出：一个文件，三元组形式：[[数字序列], [数字序列], 0或者1]
'''
import re
import sys
import random
import jieba
import pickle
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict

jieba.load_userdict("./data/pretrained_word.txt")
############################# 数据增强 #############################
def data_argumentation(train_file = "./data/train-set.data", gen_train_file = "./data/gen_train-set.data"):
    def change_sentence(sentence):
        sentence1 = [word for word in jieba.cut(sentence)]
        sentence_len = len(sentence1)

        return_sentences = []
        # 每个词语以10%的概率随机删除
        sentence2 = []
        for word in sentence1:
            if random.uniform(0, 1)>0.1:
                sentence2.append(word)
        if len(sentence2) > 0:
            sentence2 = "".join(sentence2)
            return_sentences.append(sentence2)

        # 随机选择5对词交换:
        if sentence_len>2:
            origin = list(range(sentence_len))
            for _ in range(5):
                i, j = random.sample(list(range(sentence_len)), 2)
                origin[i], origin[j] = origin[j], origin[i]
            sentence3 = [sentence1[i] for i in origin]
            sentence3 = "".join(sentence3)
            return_sentences.append(sentence3)

        # # 每个词以50%概率替换为同义词
        # sentence4 = []
        # for word in sentence1:
        #     nearby_words = synonyms.nearby(word)[0]
        #     if random.uniform(0, 1) > 0.5 and len(nearby_words) > 0:
        #         sentence4.append(random.choice(nearby_words))
        #     else:
        #         sentence4.append(word)
        # sentence4 = "".join(sentence4)
        return return_sentences

    q2pos = defaultdict(lambda: [])
    q2neg = defaultdict(lambda: [])
    all_answers = set()
    with open(train_file) as f:
        for line in f:
            line = line.strip().split("\t")
            if line[2] == "1":
                q2pos[line[0]].append(line[1])
            else:
                q2neg[line[0]].append(line[1])
            all_answers.add(line[1])
    count = 0
    for q in q2pos.keys():
        count+=1
        sys.stdout.write("%d in %d\r" % (count, len(q2pos)))
        sys.stdout.flush()
        if len(q2neg[q]) < 5:
            q2neg[q].extend(random.sample(list(all_answers - set(q2pos[q])), 5 - len(q2neg[q])))
    #     new_neg = []
    #     for neg in random.sample(q2neg[q], 15):
    #         new_neg.extend(change_sentence(neg))
    #     q2neg[q].extend(new_neg)
    #
    # q2neg_count = defaultdict(lambda: 0)
    # for q in q2pos.keys():
    #     q2neg_count[len(q2neg[q])] += 1
    # sorted(q2neg_count.items())

    with open(gen_train_file, "w") as fw:
        for q in q2pos.keys():
            for a in q2pos[q]:
                fw.write("%s\t%s\t%s\n" % (q, a, "1"))
        for q in q2neg.keys():
            for a in q2neg[q]:
                fw.write("%s\t%s\t%s\n" % (q, a, "0"))

############################# 分词部分 #############################
def get_id_sequence(train_file = "./data/gen_train-set.data", valid_file = "./data/validation-set.data",
                    new_train_file = "./data/train.pkl", new_valid_file = "./data/valid.pkl"):
    word2id = {}
    id2word = {}
    sentences = set()
    sentences2ids = {}
    # def clean_zh_text(text):
    #     # keep English, digital and Chinese
    #     comp = re.compile('[^A-Z^a-z^0-9^\u4e00-\u9fa5]')
    #     text = comp.sub(' ', text)
    #     comp = re.compile(' +')
    #     text = comp.sub(' ', text)
    #     return text

    for word in ["[PAD]", "[START]", "[END]"]:
        new_id = len(word2id)
        word2id[word] = new_id
        id2word[new_id] = word

    for filename in [train_file, valid_file]:
        with open(filename, "r") as f:
            for line in f:
                line = line.strip().split("\t")
                sentences.add(line[0])
                sentences.add(line[1])
                # sentences.add(clean_zh_text(line[0]))
                # sentences.add(clean_zh_text(line[1]))

    for sentence in tqdm(sentences):
        words = jieba.cut(sentence, cut_all=False)
        sentences2ids[sentence] = []
        sentences2ids[sentence].append(word2id["[START]"])
        for word in words:
            if word not in word2id:
                new_id = len(word2id)
                word2id[word] = new_id
                id2word[new_id] = word
            sentences2ids[sentence].append(word2id[word])
        sentences2ids[sentence].append(word2id["[END]"])

    with open("./data/word2id.pkl", "wb") as fw:
        pickle.dump(word2id, fw)

    with open("./data/id2word.pkl", "wb") as fw:
        pickle.dump(id2word, fw)

    with open(train_file, "r") as f, open(new_train_file, "wb") as fw:
        train_triples = []
        for line in f:
            line = line.strip().split("\t")
            train_triples.append((sentences2ids[line[0]], sentences2ids[line[1]], int(line[2])))
            # train_triples.append((sentences2ids[clean_zh_text(line[0])], sentences2ids[clean_zh_text(line[1])], int(line[2])))
        pickle.dump(train_triples, fw)

    with open(valid_file, "r") as f, open(new_valid_file, "wb") as fw:
        valid_triples = []
        for line in f:
            line = line.strip().split("\t")
            valid_triples.append((sentences2ids[line[0]], sentences2ids[line[1]], int(line[2])))
            # valid_triples.append((sentences2ids[clean_zh_text(line[0])], sentences2ids[clean_zh_text(line[1])], int(line[2])))
        pickle.dump(valid_triples, fw)

# def ids2word(ids):
#     return [id2word[id] for id in ids]
#
# max_que_len, max_ans_len = 0, 0
# for que, ans, label in train_triples:
#     max_que_len = max(max_que_len, len(que))
#     max_ans_len = max(max_ans_len, len(ans))
# print(f"max_que_len:{max_que_len}\t, max_ans_len:{max_ans_len}")
# for que, ans, label in valid_triples:
#     max_que_len = max(max_que_len, len(que))
#     max_ans_len = max(max_ans_len, len(ans))
# print(f"max_que_len:{max_que_len}\t, max_ans_len:{max_ans_len}")
# from collections import defaultdict
# q = defaultdict(lambda: 0)
# for que, ans, label in train_triples:
#     if label == 1 and len(ans) > 100:
#         print(ids2word(que))
#         print(ids2word(ans)[:100])
#         zhu = ids2word(ans)


############################# 词向量部分 #############################
def generate_pretrain_word_embedding(pretrain_path="./data/baidubaike", word2id_path="./data/word2id.pkl",
                                     id2word_path="./data/id2word.pkl", output_path="./data/pretrain.pt"):
    word2id = pickle.load(open(word2id_path, "rb"))
    id2word = pickle.load(open(id2word_path, "rb"))
    vocab_size = 635975
    vector_size = 300
    word_matrix = torch.zeros(len(word2id), vector_size)

    vocab = set()
    def add_word(_word, _weights):
        if _word not in word2id:
            return
        vocab.add(_word)
        word_matrix[word2id[_word]] = _weights

    with open(pretrain_path) as fin:
        for line_no, line in enumerate(fin):
            parts = line.rstrip().split(" ")
            word, weights = parts[0], list(map(float, parts[1:]))
            weights = torch.Tensor(weights)
            add_word(word, weights)

    scale = torch.std(word_matrix)
    random_range = (-scale, scale)
    random_init_count = 0
    for word in word2id.keys():
        if word not in vocab:
            random_init_count += 1
            nn.init.uniform_(word_matrix[word2id[word]], random_range[0], random_range[1])
    torch.save(word_matrix, output_path)
    print(len(word2id))
    print(random_init_count)

# pretrained_word_path="./data/pretrained_word.txt"
# pretrain_path="./data/baidubaike"
# with open(pretrain_path) as fin, open(pretrained_word_path, "w") as fout:
#     for line_no, line in enumerate(fin):
#         parts = line.rstrip().split(" ")
#         word, weights = parts[0], list(map(float, parts[1:]))
#         fout.write(word + "\n")

data_argumentation()
get_id_sequence()
generate_pretrain_word_embedding()