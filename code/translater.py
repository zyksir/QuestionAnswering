#!/usr/bin/env python
#-*- coding: utf-8 -*-
'''
输入：一个文件，三元组形式：[自然语言问题，自然语言答案，标签]
输出：一个文件，三元组形式：[[数字序列], [数字序列], 0或者1]
'''
import jieba
import pickle

train_file = ""
valid_file = ""
new_train_file = ""
new_valid_file = ""
word2id = {}
id2word = {}
sentences = set()
sentences2ids = {}

with open(train_file, "r") as f:
    for line in f:
        line = line.strip().split("\t")
        sentences.add(line[0])
        sentences.add(line[1])

with open(valid_file, "r") as f:
    for line in f:
        line = line.strip().split("\t")
        sentences.add(line[0])
        sentences.add(line[1])

for sentence in sentences:
    words = jieba.cut_for_search(sentence)
    for word in words:
        if word not in word2id:
            new_id = len(word2id)
            word2id[word] = new_id
            id2word[new_id] = word
    sentences2ids[sentence] = [word2id[word] for word in words]

with open(train_file, "r") as f, open(new_train_file, "w") as fw:
    train_triples = []
    for line in f:
        line = line.strip().split("\t")
        train_triples.append((sentences2ids[line[0]], sentences2ids[line[1]], int(line[2])))
    pickle.dump(train_triples, fw)

with open(valid_file, "r") as f, open(new_valid_file, "w") as fw:
    valid_triples = []
    for line in f:
        line = line.strip().split("\t")
        valid_triples.append((sentences2ids[line[0]], sentences2ids[line[1]], int(line[2])))
    pickle.dump(valid_triples, fw)
