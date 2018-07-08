import argparse
import os
import torch


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--name', type=str, default='res_', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 10)')
        self.parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
        self.parser.add_argument('--seed', type=int, default=0, help='initial random seed for deterministic results')
        self.parser.add_argument('--seeds', type=int, default=5, help='initial random seed for deterministic results')

        self.parser.add_argument('--lr', type=float, default=0.001, metavar='LR',help='learning rate (default: 0.01)')
        self.parser.add_argument('--momentum', type=float, default=0.5, metavar='M',help='SGD momentum (default: 0.5)')
        self.parser.add_argument('--val_size', type=int, default=10000,help='Validation datazet size')
        self.parser.add_argument('--train_size', type=int, default=0,help='Train datazet size')

        self.parser.add_argument('--first_size', type=int, default=128, help='size first fc layer')
        self.parser.add_argument('--n_hidden', type=int, default=0, help='hidden_lstm layers')
        self.parser.add_argument('--shape_lstm', type=int, default=0, help='shape lstm')
        self.parser.add_argument('--lstm', type=int, default=0, help='lstm')
        self.parser.add_argument('--drop1', type=float, default=0, help='dropout prob')
        self.parser.add_argument('--drop2', type=float, default=0.5, help='dropout prob')
        self.parser.add_argument('--one_layer', type=int, default=0, help='onle layer only')

        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--nThreads', default=8, type=int, help='# threads for loading data')
        self.parser.add_argument('--train_log_interval', type = int, default = 10,help = 'how many batches to wait before logging training status')
        self.parser.add_argument('--device', type=str, default="cuda", help='Train subsampling flag')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])


        return self.opt
