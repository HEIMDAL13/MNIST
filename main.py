from __future__ import print_function
import argparse
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from options import Options
from models import Net_2FC,Net_3FC, Net_LSTM, Net_1FC, LeNet, LeNet_LSTM, Net_RNN, LeNet_plus, Net_LSTM2, Net_3FC, Net_LSTM25, LeNet_plus_LSTM, LeNet_plus_RNN, LeNet_RNN, LeNet_GRU
from models import LeNet_LSTM_exp, LeNet_LSTM_mo, LeNet_LSTM_directconv, Net_LSTM_cell
import models
from vgg import VGG, VGG_LSTM
from data import CustomDatasetDataLoader
from solver import Solver
import numpy as np
from visualizer import Visualizer
import time
from torchvision import datasets
from tensorboardX import SummaryWriter
from collections import defaultdict
import csv

acumulated_lstm = 0
gradient_lstm_lr = 0
gradient_lstm_ud = 0
gradient_gru_lr = 0
gradient_gru_ud = 0
gradient_fc1 = 0
gradient_fc2 = 0
gradient_fc3 = 0
global_str = ""
grads = defaultdict(list)
fileg = open("gradients.txt", "w")

def main():
    global fileg
    opt = Options().parse()
    visualizer = Visualizer(opt)

    torch.manual_seed(opt.seed)
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(opt.gpu)

    data_loader = CustomDatasetDataLoader()
    data_loader.initialize(opt)
    if opt.seeds == 1:
        opt.seed = 0
        train_test(opt, visualizer, data_loader)
    else:
        average(opt,visualizer, data_loader)


def train_test(opt,visualizer,data_loader):
    global global_str
    global fileg
    fileg.close()
    fileg = open("gradients_"+opt.name+"s"+str(opt.seed)+".txt", "w")
    if opt.plot_grad==1:
        writer = SummaryWriter()

    visualizer.start_timmer()

    if opt.model=="2fc":
        model = Net_2FC(opt)
        print("Two layer model created")
    elif opt.model == "1fc":
        model = Net_1FC(opt)
        print("One layer model created")
    elif opt.model == "3fc":
        model = Net_3FC(opt)
        print("Three layer model created")
    elif opt.model == "lstm":
        model = Net_LSTM(opt)
        print("LSTM model created")
    elif opt.model=="lstm2":
        model = Net_LSTM2(opt)
        print("LSTM2 model created")
    elif opt.model == "rnn":
        model = Net_RNN(opt)
        print("RNN model created")
    elif opt.model=="lenet":
        model = LeNet(opt)
        print("Lenet model created")
    elif opt.model == "lenetlstm":
        model = LeNet_LSTM(opt)
        print("Lenet model+lstm created")
    elif opt.model == "lenet+" or opt.model == "lenetplus":
        model = LeNet_plus(opt)
        print("Lenet model+ created")
    elif opt.model == "lenet+lstm" or opt.model == "lenetpluslstm":
        model = LeNet_plus_LSTM(opt)
        print("Lenet model+LSTM created")
    elif opt.model == "lenetrnn" or opt.model == "lenetrnn":
        model = LeNet_RNN(opt)
        print("Lenet RNN created")
    elif opt.model == "lenetgru":
        model = LeNet_GRU(opt)
        print("Lenet GRU created")
    elif opt.model == "lenetlstm_exp":
        model = LeNet_LSTM_exp(opt)
        print("Lenet LSTM_exp created")
    elif opt.model == "lenetlstm_mo":
        model = LeNet_LSTM_mo(opt)
        print("Lenet LSTM_mo created")
    elif opt.model == "lstm25":
        model = Net_LSTM25(opt)
        print("LSTM25 created")
    elif opt.model == "directconv":
        model = LeNet_LSTM_directconv(opt)
        print("directconv")
    elif opt.model == "vgg":
        model = VGG(opt)
        print("VGG")
    elif opt.model == "vgglstm":
        model = VGG_LSTM(opt)
        print("VGG_LSTM")
    elif opt.model == "lstmcell":
        model = Net_LSTM_cell(opt)
        print("LSTM Cell")
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)

    model.to(opt.device)
    print("Training on: ", opt.device)
    if opt.device == "cuda":
        print("GPU: ",torch.cuda.current_device())

    if opt.plot_grad==2:
        if opt.model == "lstm":
            model.fc1.register_backward_hook(printgradnorm_fc1_e)
            model.lstm_pose_lr.register_backward_hook(printgradnorm_lstm_lr)
            model.lstm_pose_ud.register_backward_hook(printgradnorm_lstm_ud)
            model.fc3.register_backward_hook(printgradnorm_fc3)

        elif opt.model == "2fc":
            model.fc1.register_backward_hook(printgradnorm_fc1_e)
            model.fc2.register_backward_hook(printgradnorm_fc2)
            model.fc3.register_backward_hook(printgradnorm_fc3)
        elif opt.model == "lenetlstm":
            model.conv1.register_backward_hook(printgradnorm_conv1)
            model.conv2.register_backward_hook(printgradnorm_conv2)
            model.fc1.register_backward_hook(printgradnorm_fc1)
            model.lstm_pose_lr.register_backward_hook(printgradnorm_lstm_lr)
            model.lstm_pose_ud.register_backward_hook(printgradnorm_lstm_ud)
            model.fc3.register_backward_hook(printgradnorm_fc3)
        elif opt.model == "lenetgru":
            model.conv1.register_backward_hook(printgradnorm_conv1)
            model.conv2.register_backward_hook(printgradnorm_conv2)
            model.fc1.register_backward_hook(printgradnorm_fc1)
            model.gru_pose_lr.register_backward_hook(printgradnorm_gru_lr)
            model.gru_pose_ud.register_backward_hook(printgradnorm_gru_ud)
            model.fc3.register_backward_hook(printgradnorm_fc3)
        elif opt.model == "lenet":
            model.conv1.register_backward_hook(printgradnorm_conv1)
            model.conv2.register_backward_hook(printgradnorm_conv2)
            model.fc1.register_backward_hook(printgradnorm_fc1)
            model.fc2.register_backward_hook(printgradnorm_fc2)
            model.fc3.register_backward_hook(printgradnorm_fc3)

    if opt.plot_grad==3:
        for m_name, module in model.named_modules():
            print("MODULE NAME: " + m_name)
            if len(m_name)>1:
                for p_name, p in module.named_parameters():
                    print("parameter name: "+p_name)
                    p.register_hook(save_grad(m_name+"-"+p_name))

    save_net(model,visualizer.filename)
    best_epoch = 0
    solver = Solver(model, data_loader, opt,visualizer)
    best_acc = 0
    visualizer.set_filename(opt)
    visualizer.write_options()
    visualizer.model_str=str(model)
    visualizer.write_network_structure()
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    visualizer.write_text("Total Network parameters: " + str(pytorch_total_params))

    train_losses=[]
    val_losses=[]

    for epoch in range(1, opt.epochs + 1):
        fileg.write(global_str)
        fileg.write("\n EPOCH: " + str(epoch)+ "\n")
        global_str = ""
        start_epoch_time = time.time()
        train_loss,train_acc = solver.train(epoch)
        print("Accuracy TRAINING: ",train_acc)
        train_losses.append(train_loss)
        val_acc,val_loss = solver.val()
        val_losses.append(val_loss)
        if opt.display_id > 0:
            visualizer.plot_current_errors(epoch, train_loss, val_loss,train_acc,val_acc)
        if best_acc<val_acc:
            best_acc=val_acc
            save_net(model,visualizer.filename)
            best_epoch = epoch
        #
        # for name, param in model.named_parameters():
        #    if 'bn' not in name:
        #       writer.add_histogram(name, param, epoch,bins='doane')
        if opt.plot_grad==1:
            for name, param in filter(lambda np: np[1].grad is not None, model.named_parameters()):
                writer.add_histogram(name, param.grad.data.cpu().numpy(),epoch,bins='doane')

        params = list(model.parameters())

        end_epoch_time = time.time()
        epoch_time = end_epoch_time-start_epoch_time
        print("Epoch time: ",epoch_time)


    visualizer.write_text("Evaluation last model:")
    acc_last, loss_last = solver.test()
    visualizer.write_text("Evaluation best model epoch " +str(best_epoch))

    load_net(model,visualizer.filename)
    acc_best, loss_best = solver.test()
    visualizer.write_time()
    visualizer.flush_to_file()
    if opt.plot_grad==3:
        keys = sorted(grads.keys())
        with open("sgradients_"+opt.name+"s"+str(opt.seed)+".csv", 'w') as resultFile:
            wr = csv.writer(resultFile, dialect='excel')
            #wr.writerows([['apple','cherry','orange','pineapple','strawberry']])
            wr.writerow(keys)
            wr.writerows(zip(*[grads[key] for key in keys]))
    return train_losses, val_losses, acc_best, acc_last, loss_best, loss_last, best_epoch

def average(opt,visualizer,data_loader):

    torch.manual_seed(opt.seed)
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(opt.gpu)

    train_losses_av = np.zeros(opt.epochs)
    val_losses_av = np.zeros(opt.epochs)
    acc_best_v = []
    acc_last_v = []
    loss_best_v = []
    loss_last_v = []
    best_epoch_v = []


    for seed in range(0, opt.seeds):
        print("SEED no: ",seed)
        opt.seed = seed
        visualizer.reset_plot()
        train_losses, val_losses, acc_best, acc_last, loss_best, loss_last, best_epoch = train_test(opt,visualizer,data_loader)
        acc_best_v.append(acc_best)
        acc_last_v.append(acc_last)
        loss_best_v.append(loss_best)
        loss_last_v.append(loss_last)
        best_epoch_v.append(best_epoch)

        train_losses_av = train_losses_av + np.array(train_losses)
        val_losses_av = val_losses_av + np.array(val_losses)

    train_losses_av /= opt.seeds
    val_losses_av /= opt.seeds
    acc_best_m = np.mean(np.array(acc_best_v))
    acc_last_m = np.mean(np.array(acc_last_v))
    loss_best_m = np.mean(np.array(loss_best_v))
    loss_last_m = np.mean(np.array(loss_last_v))
    best_epoch_m = np.mean(np.array(best_epoch_v))
    acc_best_std = np.std(np.array(acc_best_v))
    acc_last_std = np.std(np.array(acc_last_v))
    loss_best_std = np.std(np.array(loss_best_v))
    loss_last_std = np.std(np.array(loss_last_v))
    best_epoch_std = np.std(np.array(best_epoch_v))

    visualizer.set_filename_av(opt)
    visualizer.write_options()
    visualizer.write_network_structure()
    visualizer.write_text("\nBest Model: acc: {:.2f}±{:.2f}".format(acc_best_m, acc_best_std))
    visualizer.write_text("Last Model: acc: {:.2f}±{:.2f}".format(acc_last_m, acc_last_std))
    visualizer.write_text("Best epoch: {:.2f}±{:.2f} \n".format(best_epoch_m, best_epoch_std))
    visualizer.write_text("Best Model: loss: {:.2f}±{:.2f}".format(loss_best_m, loss_best_std))
    visualizer.write_text("Last Model: loss: {:.2f}±{:.2f}".format(loss_last_m, loss_last_std))
    visualizer.flush_to_file()
    return acc_last_m

def save_net(net,filename):
    save_filename = 'net_'+filename+'.pth'
    save_path = './results/'
    torch.save(net.state_dict(), save_path + save_filename)

def load_net(net,filename):
    save_filename = 'net_'+filename+'.pth'
    save_path = './results/'
    net.load_state_dict(torch.load(save_path + save_filename))


def printnorm(self, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('')
    print('input: ', type(input))
    print('input[0]: ', type(input[0]))
    print('output: ', type(output))
    print('')
    print('input size:', input[0].size())
    print('output size:', output.data.size())
    print('output norm:', output.data.norm())

def printgradnorm_lstm_lr(self, grad_input, grad_output):
    global gradient_lstm_lr
    global fileg
    global global_str
    print('Inside class:' + self.__class__.__name__)
    print('grad_output LSTM:', grad_output[1].norm())
    gradient_lstm_lr = grad_output[1].norm().item()
    acumulated_lstm = np.sqrt(gradient_lstm_lr*gradient_lstm_lr +gradient_lstm_ud*gradient_lstm_ud)
    print("Acumulated lstm: ", acumulated_lstm)
    global_str += "(" + str(acumulated_lstm)+ ") LSTM -> "
    print("GLOBAL STR: " + global_str)
    print("2")


def printgradnorm_lstm_ud(self, grad_input, grad_output):
    global gradient_lstm_ud
    print('Inside class:' + self.__class__.__name__)
    print('grad_output LSTM:', grad_output[1].norm())
    gradient_lstm_ud = grad_output[1].norm().item()
    print("1")


def printgradnorm_gru_lr(self, grad_input, grad_output):
    global gradient_gru_lr
    global fileg
    global global_str
    print('Inside class:' + self.__class__.__name__)
    print('grad_output LSTM:', grad_output[1].norm())
    gradient_gru_lr = grad_output[1].norm().item()
    acumulated_lstm = np.sqrt(gradient_gru_lr*gradient_gru_lr + gradient_gru_ud*gradient_gru_ud)
    print("Acumulated lstm: ", acumulated_lstm)
    global_str += "(" + str(acumulated_lstm)+ ") GRU -> "
    print("GLOBAL STR: " + global_str)
    print("2")


def printgradnorm_gru_ud(self, grad_input, grad_output):
    global gradient_gru_ud
    print('Inside class:' + self.__class__.__name__)
    print('grad_output LSTM:', grad_output[1].norm())
    gradient_gru_ud = grad_output[1].norm().item()
    print("1")

def printgradnorm_fc1_e(self, grad_input, grad_output):
    global global_str
    print('Inside class:' + self.__class__.__name__)
    gradient_fc1 = grad_output[0].norm().item()
    print('grad_output fc1:', grad_output[0].norm())
    global_str += "(" + str(gradient_fc1)+ ") FC1 \n"



def printgradnorm_fc1(self, grad_input, grad_output):
    global global_str
    print('Inside class:' + self.__class__.__name__)
    gradient_fc1 = grad_output[0].norm().item()
    print('grad_output fc1:', grad_output[0].norm())
    global_str += "(" + str(gradient_fc1)+ ") FC1 ->"

def printgradnorm_fc2(self, grad_input, grad_output):
    global global_str
    print('Inside class:' + self.__class__.__name__)
    gradient_fc2 = grad_output[0].norm().item()
    print('grad_output fc2:', grad_output[0].norm())
    global_str += "(" + str(gradient_fc2)+ ") FC2 ->"

def printgradnorm_fc3(self, grad_input, grad_output):
    global global_str
    print('Inside class:' + self.__class__.__name__)
    gradient_fc3 = grad_output[0].norm().item()
    print('grad_output fc3:', gradient_fc3)
    global_str += "(" + str(gradient_fc3)+ ") FC3 ->"

def printgradnorm_conv1(self, grad_input, grad_output):
    global global_str
    print('Inside class:' + self.__class__.__name__)
    gradient_conv1 = grad_output[0].norm().item()
    print('grad_output fc3:', gradient_conv1)
    global_str += "(" + str(gradient_conv1)+ ") conv1\n"

def printgradnorm_conv2(self, grad_input, grad_output):
    global global_str
    print('Inside class:' + self.__class__.__name__)
    gradient_conv2 = grad_output[0].norm().item()
    print('grad_output fc3:', gradient_conv2)
    global_str += "(" + str(gradient_conv2)+ ") conv2 ->"

def save_grad(name):
    def hook(grad):
        print(name)
        if "lstm" in name:
            grad_i = grad[0:256].data.norm().item()
            grads[name+"_i"].append(grad_i)
            grad_f = grad[256:512].data.norm().item()
            grads[name + "_f"].append(grad_f)
            grad_g = grad[512:768].data.norm().item()
            grads[name + "_g"].append(grad_g)
            grad_o = grad[768:1024].data.norm().item()
            grads[name + "_o"].append(grad_o)
        else:
            grads[name].append(grad.data.norm().item())
    return hook



if __name__ == '__main__':
    main()