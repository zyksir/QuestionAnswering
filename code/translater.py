#!/usr/bin/env python
#-*- coding: utf-8 -*-
'''
输入：一个文件，三元组形式：[自然语言问题，自然语言答案，标签]
输出：一个文件，三元组形式：[[数字序列], [数字序列], 0或者1]
'''
import re
import jieba
import pickle
from tqdm import tqdm

train_file = "./data/train-set.data"
valid_file = "./data/validation-set.data"
new_train_file = "./data/train.pkl"
new_valid_file = "./data/valid.pkl"
word2id = {}
id2word = {}
sentences = set()
sentences2ids = {}

def clean_zh_text(text):
    # keep English, digital and Chinese
    comp = re.compile('[^A-Z^a-z^0-9^\u4e00-\u9fa5]')
    text = comp.sub(' ', text)
    comp = re.compile(' +')
    text = comp.sub(' ', text)
    return text

for word in ["[PAD]", "[START]", "[END]"]:
    new_id = len(word2id)
    word2id[word] = new_id
    id2word[new_id] = word

with open(train_file, "r") as f:
    for line in f:
        line = line.strip().split("\t")
        sentences.add(clean_zh_text(line[0]))
        sentences.add(clean_zh_text(line[1]))

with open(valid_file, "r") as f:
    for line in f:
        line = line.strip().split("\t")
        sentences.add(clean_zh_text(line[0]))
        sentences.add(clean_zh_text(line[1]))

for sentence in tqdm(sentences):
    words = jieba.cut(sentence, cut_all=False)
    sentences2ids[sentence] = []
    for word in words:
        if word == " ":
            continue
        if word not in word2id:
            new_id = len(word2id)
            word2id[word] = new_id
            id2word[new_id] = word
        sentences2ids[sentence].append(word2id[word])

with open("./data/word2id.pkl", "wb") as fw:
    pickle.dump(word2id, fw)

with open("./data/id2word.pkl", "wb") as fw:
    pickle.dump(id2word, fw)

with open(train_file, "r") as f, open(new_train_file, "wb") as fw:
    train_triples = []
    for line in f:
        line = line.strip().split("\t")
        train_triples.append((sentences2ids[clean_zh_text(line[0])], sentences2ids[clean_zh_text(line[1])], int(line[2])))
    pickle.dump(train_triples, fw)

with open(valid_file, "r") as f, open(new_valid_file, "wb") as fw:
    valid_triples = []
    for line in f:
        line = line.strip().split("\t")
        valid_triples.append((sentences2ids[clean_zh_text(line[0])], sentences2ids[clean_zh_text(line[1])], int(line[2])))
    pickle.dump(valid_triples, fw)

def ids2word(ids):
    return [id2word[id] for id in ids]

max_que_len, max_ans_len = 0, 0
for que, ans, label in train_triples:
    max_que_len = max(max_que_len, len(que))
    max_ans_len = max(max_ans_len, len(ans))
print(f"max_que_len:{max_que_len}\t, max_ans_len:{max_ans_len}")
for que, ans, label in valid_triples:
    max_que_len = max(max_que_len, len(que))
    max_ans_len = max(max_ans_len, len(ans))
print(f"max_que_len:{max_que_len}\t, max_ans_len:{max_ans_len}")
from collections import defaultdict
q = defaultdict(lambda: 0)
for que, ans, label in train_triples:
    if label == 1 and len(ans) > 100:
        print(ids2word(que))
        print(ids2word(ans)[:100])
        zhu = ids2word(ans)