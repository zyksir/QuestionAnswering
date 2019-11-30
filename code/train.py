#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os
import torch
import logging
from torch.utils.data import DataLoader
import argparse
from model import CNNModel
from dataloader import TrainDataset

torch.set_num_threads(8)

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing QuestionAnswering Models',
        usage='train.py [<args>] [-h | --help]'
    )
    parser.add_argument('--train_file', type=str, default="./data/train.pkl")
    parser.add_argument('--valid_file', type=str, default="./data/vaild.pkl")
    parser.add_argument('--negative_sample_size', type=int, default=1)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float)
    parser.add_argument('-save', '--save_path', default="./models/", type=str)

    parser.add_argument('--word_num', default=323297, type=int)
    parser.add_argument('--embed_dim', default=1000, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)

    parser.add_argument('--channel_size', type=int, default=8)
    parser.add_argument('--max_que_len', type=int, default=50)
    parser.add_argument('--max_ans_len', type=int, default=50)





    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('--gen_init', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)

    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    return parser.parse_args(args)

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    log_file = os.path.join(args.save_path, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def main(args):
    #### build dataloader and logger
    set_logger(args)
    train_dataset = TrainDataset(args.train_file, args.negative_sample_size)
    valid_dataset = TrainDataset(args.valid_file)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=max(1, args.cpu_num // 2),
        collate_fn=TrainDataset.collate_fn
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=max(1, args.cpu_num // 2),
        collate_fn=TrainDataset.collate_fn
    )

    #### build model
    model = CNNModel(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if args.cuda:
        model = model.cuda()

    #### begin training
    for epoch in range(args.num_epochs):
        model.do_train(model, optimizer, train_dataloader, args)
        if (epoch+1) % args.valid_steps:
            model.do_valid(model, valid_dataloader, args)


if __name__ == '__main__':
    main(parse_args())