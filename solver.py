import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import copy
from torch.autograd import Variable
import itertools

class Solver(object):
    """Neural Network Training Solver."""

    def __init__(self, model, data_loader, opt, visualizer):
        self._lr = opt.lr
        self._epochs = opt.epochs
        self._train_log_interval = opt.train_log_interval
        self._model = model
        self._data_loader = data_loader
        self._momentum = opt.momentum
        self._optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum,weight_decay=0.0002)
        #self._optimizer = optim.Adam(model.parameters(), lr=opt.lr, eps=1, betas=(0.9, 0.999))
        self._train_loader = self._data_loader.get_train_loader()
        self._val_loader = self._data_loader.get_val_loader()
        self._test_loader = self._data_loader.get_test_loader()
        self.visualizer = visualizer
        self._device = opt.device


    def train(self,epoch):
        """Train model specified epochs on data_loader."""
        self._model.train()
        av_loss = 0
        for batch_idx, (data, target) in enumerate(self._train_loader):
            data, target = data.to(self._device), target.to(self._device)
            self._optimizer.zero_grad()
            output = self._model(data)
            loss = F.nll_loss(output, target)
            av_loss+=loss.item()
            loss.backward()
            self._optimizer.step()
            if batch_idx % self._train_log_interval == 0:
                self.visualizer.write_text('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self._train_loader.sampler),
                           100. * batch_idx / len(self._train_loader), loss.item()))
        av_loss /= batch_idx
        self.visualizer.write_text('Train Average loss: {:.4f}'.format(av_loss))

        return av_loss

    def test(self, model=None):
        if model is None:
            model = self._model
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self._test_loader:
                data, target = data.to(self._device), target.to(self._device)
                output = model(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self._test_loader.dataset)
        test_acc = 100. * correct / len(self._test_loader.dataset)
        self.visualizer.write_text('\nTest set Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(self._test_loader.dataset), test_acc))

        return test_acc,test_loss


    def val(self):
        model = self._model
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self._val_loader:
                data, target = data.to(self._device), target.to(self._device)
                output = model(data)
                val_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(self._val_loader.sampler)
        val_acc = 100. * correct / len(self._val_loader.sampler)
        self.visualizer.write_text('\nValidation set Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            val_loss, correct, len(self._val_loader.sampler),
            100. * correct / len(self._val_loader.sampler)))

        return val_acc, val_loss

    def check_dataset(self):
        i = 0
        same = 0
        for data_val, target in self._val_loader:
            i+=1
            if i % self._train_log_interval == 0:
                print("Checking validation image ",i)
            for data_train, target in self._train_loader:
                if torch.eq(data_val, data_train).all():
                    print("Train and validation use same images!!!")
                    same+=1
        print("finished! Everythink ok")

        return same