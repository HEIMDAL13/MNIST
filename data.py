import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from subset_sampler import SubsetSampler
import numpy as np

class CustomDatasetDataLoader():

    def initialize(self, opt):

        if opt.dataset == "mnist":
            print("MNIST dataset laoded!")

            self.train_dataset = datasets.MNIST('../data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))
            self.test_dataset = datasets.MNIST('../data', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))
        elif opt.dataset == "cifar10":
            print("Cifar10 dataset laoded!")
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            self.train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

            self.test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        else:
            raise ValueError("Dataset [%s] not recognized." % opt.dataset)

        self.test_loader = torch.utils.data.DataLoader(self.test_dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers= 1,
            pin_memory=True)

        train_idx = torch.arange(len(self.train_dataset)).long()
        if opt.val_size:
            train_idx, val_idx = train_idx[opt.val_size:], train_idx[:opt.val_size]
            val_sampler = SubsetRandomSampler(val_idx)
            self.val_loader = torch.utils.data.DataLoader(self.train_dataset,
                                    sampler=val_sampler,
                                    batch_size=opt.batch_size,
                                    shuffle=False)
        else:
            self.val_loader = self.test_loader

        # train_sampler = SubsetRandomSampler(train_idx)

        # set train subset size
        train_subset_size = opt.train_size
        # sample subset from train part of previously split train dataset

        train_sampler = SubsetSampler(train_idx,
                                          train_subset_size,
                                          random_subset=True,
                                          shuffle=True)


        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                  sampler=train_sampler,
                                  batch_size=opt.batch_size,
                                  num_workers=1,
                                  shuffle=False)

    def get_train_loader(self):
        return self.train_loader
    def get_val_loader(self):
        return self.val_loader
    def get_test_loader(self):
        return self.test_loader


