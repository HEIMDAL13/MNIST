import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import OrderedDict
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import visdom


class Visualizer():

    def __init__(self,opt):
        self.filename = ""
        print(time.strftime("%Y-%m-%d %H:%M"))
        self.text = time.strftime("%Y-%m-%d %H:%M")
        self.opt = opt
        self.start_time = 0
        self.end_time = 0
        self.display_id = opt.display_id*10
        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom(port=opt.display_port)
        if not os.path.exists("results"):
            os.makedirs("results")
        if opt.vis_env == "default":
            self.env = str(opt.momentum)+'_'+opt.model+'_drop'+str(0!=opt.drop2)+'_'+ str(opt.train_size)
        else:
            self.env = opt.vis_env
        print("SELF ENV: ",self.env)

    def write_network_structure(self):
        self.write_text("Network Structure: \n")

        if self.opt.model == "2fc":
            self.write_text("1: FC 784x" + str(self.opt.first_size))
            self.write_text("2: FC " + str(self.opt.first_size) + "x" + str(self.opt.n_hidden * 4))
            self.write_text("3: FC " + str(self.opt.n_hidden * 4) + "x10\n")
        elif self.opt.model == "lstm":
            self.write_text("1: FC 784x" + str(self.opt.first_size))
            if self.opt.shape_lstm == 0:
                self.write_text("2: LSTM " + str(self.opt.first_size) + "x" + str(self.opt.n_hidden*4) + " Shaped as:" + str(
                self.opt.first_size) + "x1" + " n_hidden: " + str(self.opt.n_hidden*2))
            else:
                self.write_text("2: LSTM " + str(self.opt.first_size) + "x" + str(self.opt.n_hidden * 4) + " Shaped as:" + str(
                self.opt.shape_lstm) + "x" + str(int(self.opt.first_size / self.opt.shape_lstm)) + " n_hidden: " + str(
                self.opt.n_hidden))
            self.write_text("3: FC " + str(self.opt.n_hidden * 4)+ "x10\n")
        elif self.opt.model == "fc":
            self.write_text("1: FC 784x" + str(self.opt.first_size))
            self.write_text("2: FC " + str(self.opt.n_hidden * 4) + "x10\n")

    def write_val_result(self, in_text):
        self.text+="sd"

    def write_options(self):
        argx = vars(self.opt)
        self.write_text('\n------------ Options -------------')
        for k, v in sorted(argx.items()):
            self.write_text('%s: %s' % (str(k), str(v)))
        self.write_text('-------------- End ----------------\n\n')
    def write_num_parameters(self,model):
        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        self.write_text("Total Network parameters: " + str(pytorch_total_params))

    def write_test_resut(self, in_text):
        self.text+="sd"

    def write_text(self, in_text):
        print(in_text)
        self.text+=in_text+"\n"

    def write_time(self):
        exec_time = self.stop_timmer()
        self.write_text("Total execution time: " + time.strftime("%H:%M:%S", time.gmtime(exec_time)))

    def plot_trainval(self,train,val):
        plt.plot(np.arange(1,len(train)+1),train,np.arange(1,len(val)+1),val)
        plt.ylabel('loss')
        plt.title('train/test loss')
        plt.savefig('./results/plot_' + self.filename + '.jpg')
        plt.close()

    def flush_to_file(self):
        file = open("./results/"+self.filename+".txt", "w")
        file.write(self.text)
        self.text=""
        file.close()

    def set_filename(self,opt):
        if opt.name =="":
            self.filename = str(opt.model)+str(opt.first_size)+"x"+str(opt.n_hidden*4)+'drop'+str(int(opt.drop1*100))+"x"+ str(int(opt.drop2*100)) + "smpl" + str(opt.train_size)+"seed"+str(opt.seed)
        else:
            self.filename = opt.name+str(opt.seed)

    def set_filename_av(self,opt):
        if opt.name =="":
            self.filename = "AVERAGE_"+str(opt.model)+str(opt.first_size)+"x"+str(opt.n_hidden*4)+'drop'+str(int(opt.drop1*100))+"x"+ str(int(opt.drop2*100)) + "smpl" + str(opt.train_size)
        else:
            self.filename = "AVERAGE_"+opt.name


    def start_timmer(self):
        self.start_time = time.time()
    def stop_timmer(self):
        self.end_time = time.time()
        return self.end_time-self.start_time



    def plot_current_errors(self, epoch, train_loss, val_loss,train_acc,val_acc):
        errors = [train_loss, val_loss, train_acc,val_acc]
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': ["train_loss", "val_loss","train_acc","val_acc"]}
        self.plot_data['X'].append(epoch)
        self.plot_data['Y'].append(errors)
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.filename + ' id:' +str(self.display_id),
                'xlabel': 'epoch',
                'legend': self.plot_data['legend'],
                'ylabel': 'loss'},
            win=self.display_id,env=self.env)

    def reset_plot(self):
        self.plot_data = {'X': [], 'Y': [], 'legend': ["train_loss", "val_loss","train_acc","val_acc"]}
        self.display_id+=1