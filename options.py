import argparse
import os
import torch


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--name', type=str, default="", help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 10)')
        self.parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
        self.parser.add_argument('--seed', type=int, default=0, help='initial random seed for deterministic results')
        self.parser.add_argument('--seeds', type=int, default=5, help='initial random seed for deterministic results')

        self.parser.add_argument('--lr', type=float, default=0.001, metavar='LR',help='learning rate (default: 0.01)')
        self.parser.add_argument('--weight_decay', type=float, default=0.0002, metavar='WD',help='weight decay')
        self.parser.add_argument('--optimizer', type=str, default='SGD', metavar='OP',help='optimizer')
        self.parser.add_argument('--momentum', type=float, default=0.5, metavar='M',help='SGD momentum (default: 0.5)')
        self.parser.add_argument('--val_size', type=int, default=10000,help='Validation datazet size')
        self.parser.add_argument('--train_size', type=int, default=0,help='Train datazet size')

        self.parser.add_argument('--model', type=str, default='2fc', help='Model to be used (2fc, lstm, 1fc)')
        self.parser.add_argument('--first_size', type=int, default=128, help='size first fc layer')
        self.parser.add_argument('--n_hidden', type=int, default=0, help='hidden_lstm layers')
        self.parser.add_argument('--shape_lstm', type=int, default=0, help='shape lstm')
        self.parser.add_argument('--drop1', type=float, default=0, help='dropout prob')
        self.parser.add_argument('--drop2', type=float, default=0.5, help='dropout prob')

        self.parser.add_argument('--gpu', type=int, default=0, help='gpu id')
        self.parser.add_argument('--nThreads', default=8, type=int, help='# threads for loading data')
        self.parser.add_argument('--train_log_interval', type = int, default = 10,help = 'how many batches to wait before logging training status')
        self.parser.add_argument('--device', type=str, default="cuda", help='Where to train the network, cuda or cpu')

        self.parser.add_argument('--display_id', type=int, default="0", help='Where to train the network, cuda or cpu')
        self.parser.add_argument('--display_port', type=int, default="8095", help='Where to train the network, cuda or cpu')
        self.parser.add_argument('--vis_env', type=str, default="default", help='Where to train the network, cuda or cpu')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt
