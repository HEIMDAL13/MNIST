import torch.nn as nn
import torch.nn.functional as F
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import os

class Visualizer():

    def __init__(self,opt):

        self.filename = opt.name+str(opt.first_size)+"x"+str(opt.n_hidden*4)+'lstm'+str(opt.lstm)+'drop'+str(opt.drop1)+"-"+ str(opt.drop2)+"seed"+str(opt.seed) + "subs" + str(opt.train_size)
        print(time.strftime("%Y-%m-%d %H:%M"))
        self.text = time.strftime("%Y-%m-%d %H:%M")
        self.opt = opt
        if not os.path.exists("results"):
            os.makedirs("results")


    def write_network_structure(self):

        if self.opt.lstm == 0:
            self.write_text("PARAMETERS: \n")
            self.write_text("1: FC 784x" + str(self.opt.first_size))
            self.write_text("2: FC " + str(self.opt.first_size) + "x" + str(self.opt.n_hidden * 4))
            self.write_text("3: FC " + str(self.opt.n_hidden * 4) + "x10")

        else:
            self.write_text("1: FC 784x" + str(self.opt.first_size))

            if self.opt.shape_lstm == 0:
                self.write_text("2: LSTM " + str(self.opt.first_size) + "x" + str(self.opt.n_hidden * 4) + " Shaped as:" + str(
                self.opt.first_size) + "x 1" + " n_hidden" + str(self.opt.n_hidden * 2))
            else:
                self.write_text("2: LSTM " + str(self.opt.first_size) + "x" + str(self.opt.n_hidden * 4) + " Shaped as:" + str(
                self.opt.shape_lstm) + "x" + str(int(self.opt.first_size / self.opt.shape_lstm)) + " n_hidden" + str(
                self.opt.n_hidden))
                self.write_text("3: FC " + str(self.opt.n_hidden * 4)+"\n\n")

    def write_val_result(self, in_text):
        self.text+="sd"

    def write_options(self):
        argx = vars(self.opt)
        self.write_text('\n------------ Options -------------')
        for k, v in sorted(argx.items()):
            self.write_text('%s: %s' % (str(k), str(v)))
        self.write_text('-------------- End ----------------\n\n')

    def write_test_resut(self, in_text):
        self.text+="sd"

    def write_text(self, in_text):
        print(in_text)
        self.text+=in_text+"\n"

    def plot_trainval(self,train,val):
        plt.plot(np.arange(1,len(train)+1),train,np.arange(1,len(val)+1),val)
        plt.ylabel('loss')
        plt.title('train/test loss')
        plt.savefig('./results/plot_' + self.filename + '.jpg')
        plt.close()
    def flush_to_file(self):
        file = open("./results/"+self.filename+".txt", "w")
        file.write(self.text)
        file.close()