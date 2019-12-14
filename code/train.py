#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os
import pickle
import torch
import logging
from torch.utils.data import DataLoader
import argparse
import pytorch_warmup as warmup
from IPython import embed
from model import CNNModel, RNNModel, CNNRNNModel
from dataloader import TrainDataset, TestDataset

torch.set_num_threads(8)
model_name2model = {
    "RNN": RNNModel,
    "CNN": CNNModel,
    "CNNRNN": CNNRNNModel
}

def log_metrics(epoch, metrics):
    for metric in metrics:
        logging.info('%s at epoch %d: %f' % (metric, epoch, metrics[metric]))

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing QuestionAnswering Models',
        usage='train.py [<args>] [-h | --help]'
    )
    parser.add_argument('--adv_temperature', type=int, default=10)
    parser.add_argument('--margin', type=float, default=0.75)
    parser.add_argument('--class_weights', type=str, default="[0.1, 1]")
    parser.add_argument('--word2id', type=str, default="./data/word2id.pkl")
    parser.add_argument('--id2word', type=str, default="./data/id2word.pkl")
    parser.add_argument('--train_file', type=str, default="./data/train.pkl")
    parser.add_argument('--valid_file', type=str, default="./data/valid.pkl")
    parser.add_argument('--negative_sample_size', type=int, default=3)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('--mode', type=str, default="UpSampling")
    parser.add_argument('--cuda', action='store_true', help='use GPU')
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('-lr', '--learning_rate', default=0.01, type=float)
    parser.add_argument('-save', '--save_path', default="./models/", type=str)

    parser.add_argument('--word_num', default=None, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)

    parser.add_argument('--channel_size', type=int, default=8)
    parser.add_argument('--que_max_len', type=int, default=50)
    parser.add_argument('--ans_max_len', type=int, default=50)
    parser.add_argument('--conv_kernel_1', type=int, default=3)
    parser.add_argument('--conv_kernel_2', type=int, default=3)

    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--birnn', type=bool, default=False)

    parser.add_argument('--model_name', type=str, default="RNN")
    parser.add_argument('--name', type=str, default="train")
    parser.add_argument('--log_step', type=int, default=50)
    parser.add_argument('--valid_epochs', type=int, default=1)
    parser.add_argument('--pretrain', type=str, default="./data/pretrain.pt")
    return parser.parse_args(args)

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    log_file = os.path.join(args.save_path, '%s.log' % args.name)

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
    if args.word_num is None:
        args.word_num = len(pickle.load(open(args.word2id, "rb")))
    set_logger(args)
    logging.info(args)
    train_dataset = TrainDataset(args.train_file, args.negative_sample_size)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=max(1, args.cpu_num // 2), collate_fn=TrainDataset.collate_fn)
    train_dataset_neg = TrainDataset(args.train_file, 1, if_pos=False)
    train_dataloader_neg = DataLoader(train_dataset_neg, batch_size=args.batch_size, shuffle=True,
                                  num_workers=max(1, args.cpu_num // 2), collate_fn=TrainDataset.collate_fn)

    valid_dataset = TestDataset(args.valid_file)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=True,
                                  num_workers=max(1, args.cpu_num // 2), collate_fn=TestDataset.collate_fn)

    #### build model
    model = model_name2model[args.model_name](args)
    if args.pretrain is not None and os.path.exists(args.pretrain):
        logging.info("using %s as pretrain word embedding" % args.pretrain)
        pretrained = torch.load(args.pretrain)
        model.embedding.weight.data.copy_(pretrained)

    num_steps = len(train_dataloader) * args.num_epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, num_steps // 2, gamma=0.1)
    warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    if args.cuda:
        model = model.cuda()

    # embed()
    #### begin training
    logging.info("begin training:")
    best_model, best_MRR = None, 0
    for epoch in range(args.num_epochs):
        model.do_train(model, optimizer, train_dataloader, args, lr_scheduler, warmup_scheduler)
        # model.do_train(model, optimizer, train_dataloader_neg, args)
        if epoch % model.config.valid_epochs == 0:
            metric = model.do_valid(model, valid_dataloader, args)
            log_metrics(epoch, metric)
            if metric["MRR"] > best_MRR:
                best_model = model.state_dict()
                best_MRR = metric["MRR"]
    torch.save(best_model, os.path.join(args.save_path, "best_%s.pt" % args.name))



if __name__ == '__main__':
    main(parse_args())