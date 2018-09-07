from __future__ import print_function
import argparse
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from options import Options
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
grads = defaultdict(list)
grads_m = defaultdict(list)


def main():
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
    torch.manual_seed(opt.seed)
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    if opt.plot_grad==1:
        writer = SummaryWriter()

    visualizer.start_timmer()
    model_class = getattr(models,opt.model)
    model = model_class(opt)
    model.to(opt.device)
    print("Training on: ", opt.device)
    if opt.device == "cuda":
        print("GPU: ",torch.cuda.current_device())

    if opt.plot_grad==2:
        for m_name, module in model.named_modules():
            print("MODULE NAME: " + m_name)
            module.register_backward_hook(save_grad_global(m_name))

    if opt.plot_grad==3:
        for m_name, module in model.named_modules():
            print("MODULE NAME: " + m_name)
            if len(m_name)>1:
                for p_name, p in module.named_parameters():
                    print("parameter name: "+p_name)
                    p.register_hook(save_grad(m_name+"-"+p_name))

    save_net(model,visualizer.filename,opt.resdir)
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
            save_net(model,visualizer.filename,opt.resdir)
            best_epoch = epoch

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

    load_net(model,visualizer.filename,opt.resdir)
    acc_best, loss_best = solver.test()
    visualizer.write_time()
    visualizer.flush_to_file()

    if opt.plot_grad>1:
        keys = sorted(grads.keys())
        with open("./"+opt.resdir+"/sgradients_"+opt.name+"s"+str(opt.seed)+".csv", 'w') as resultFile:
            wr = csv.writer(resultFile, dialect='excel')
            wr.writerow(keys)
            wr.writerows(zip(*[grads[key] for key in keys]))
        if opt.plot_grad > 2:
            final_grads = {}
            final_grads_m = {}
            lstm_flag = 0
            for key, value in grads.items():
                if (('lstm' not in key) and ("weight" in key)):
                    final_grads[key] = value
                    final_grads_m[key] = grads_m[key]
                elif "lstm" in key:
                    lstm_flag = 1
            if lstm_flag:
                final_grads["lstm input gate"] = np.sqrt(np.square(grads["lstm_pose_lr-weight_hh_l0_i"])+np.square(grads["lstm_pose_lr-weight_ih_l0_i"])+np.square(grads["lstm_pose_lr-weight_hh_l0_reverse_i"])+np.square(grads["lstm_pose_lr-weight_ih_l0_reverse_i"])+np.square(grads["lstm_pose_ud-weight_hh_l0_i"])+np.square(grads["lstm_pose_ud-weight_ih_l0_i"])+np.square(grads["lstm_pose_ud-weight_hh_l0_reverse_i"])+np.square(grads["lstm_pose_ud-weight_ih_l0_reverse_i"]))
                final_grads_m["lstm input gate"] = np.mean([grads_m["lstm_pose_lr-weight_hh_l0_i"], grads_m["lstm_pose_lr-weight_ih_l0_i"], grads_m["lstm_pose_lr-weight_hh_l0_reverse_i"], grads_m["lstm_pose_lr-weight_ih_l0_reverse_i"], grads_m["lstm_pose_ud-weight_hh_l0_i"], grads_m["lstm_pose_ud-weight_ih_l0_i"], grads_m["lstm_pose_ud-weight_hh_l0_reverse_i"], grads_m["lstm_pose_ud-weight_ih_l0_reverse_i"]],axis=0)
                final_grads["lstm output gate"] = np.sqrt(np.square(grads["lstm_pose_lr-weight_hh_l0_o"])+np.square(grads["lstm_pose_lr-weight_ih_l0_o"])+np.square(grads["lstm_pose_lr-weight_hh_l0_reverse_o"])+np.square(grads["lstm_pose_lr-weight_ih_l0_reverse_o"])+np.square(grads["lstm_pose_ud-weight_hh_l0_o"])+np.square(grads["lstm_pose_ud-weight_ih_l0_o"])+np.square(grads["lstm_pose_ud-weight_hh_l0_reverse_o"])+np.square(grads["lstm_pose_ud-weight_ih_l0_reverse_o"]))
                final_grads_m["lstm output gate"] = np.mean([grads_m["lstm_pose_lr-weight_hh_l0_o"], grads_m["lstm_pose_lr-weight_ih_l0_o"], grads_m["lstm_pose_lr-weight_hh_l0_reverse_o"], grads_m["lstm_pose_lr-weight_ih_l0_reverse_o"], grads_m["lstm_pose_ud-weight_hh_l0_o"], grads_m["lstm_pose_ud-weight_ih_l0_o"], grads_m["lstm_pose_ud-weight_hh_l0_reverse_o"], grads_m["lstm_pose_ud-weight_ih_l0_reverse_o"]],axis=0)
                final_grads["lstm forget gate"] = np.sqrt(np.square(grads["lstm_pose_lr-weight_hh_l0_f"])+np.square(grads["lstm_pose_lr-weight_ih_l0_f"])+np.square(grads["lstm_pose_lr-weight_hh_l0_reverse_f"])+np.square(grads["lstm_pose_lr-weight_ih_l0_reverse_f"])+np.square(grads["lstm_pose_ud-weight_hh_l0_f"])+np.square(grads["lstm_pose_ud-weight_ih_l0_f"])+np.square(grads["lstm_pose_ud-weight_hh_l0_reverse_f"])+np.square(grads["lstm_pose_ud-weight_ih_l0_reverse_f"]))
                final_grads_m["lstm forget gate"] = np.mean([grads_m["lstm_pose_lr-weight_hh_l0_f"], grads_m["lstm_pose_lr-weight_ih_l0_f"], grads_m["lstm_pose_lr-weight_hh_l0_reverse_f"], grads_m["lstm_pose_lr-weight_ih_l0_reverse_f"], grads_m["lstm_pose_ud-weight_hh_l0_f"], grads_m["lstm_pose_ud-weight_ih_l0_f"], grads_m["lstm_pose_ud-weight_hh_l0_reverse_f"], grads_m["lstm_pose_ud-weight_ih_l0_reverse_f"]],axis=0)
                final_grads["lstm activation"] = np.sqrt(np.square(grads["lstm_pose_lr-weight_hh_l0_g"])+np.square(grads["lstm_pose_lr-weight_ih_l0_g"])+np.square(grads["lstm_pose_lr-weight_hh_l0_reverse_g"])+np.square(grads["lstm_pose_lr-weight_ih_l0_reverse_g"])+np.square(grads["lstm_pose_ud-weight_hh_l0_g"])+np.square(grads["lstm_pose_ud-weight_ih_l0_g"])+np.square(grads["lstm_pose_ud-weight_hh_l0_reverse_g"])+np.square(grads["lstm_pose_ud-weight_ih_l0_reverse_g"]))
                final_grads_m["lstm activation"] = np.mean([grads_m["lstm_pose_lr-weight_hh_l0_g"], grads_m["lstm_pose_lr-weight_ih_l0_g"], grads_m["lstm_pose_lr-weight_hh_l0_reverse_g"], grads_m["lstm_pose_lr-weight_ih_l0_reverse_g"], grads_m["lstm_pose_ud-weight_hh_l0_g"], grads_m["lstm_pose_ud-weight_ih_l0_g"], grads_m["lstm_pose_ud-weight_hh_l0_reverse_g"], grads_m["lstm_pose_ud-weight_ih_l0_reverse_g"]],axis=0)
            keys = sorted(final_grads.keys())
            with open("./"+opt.resdir+"/sglite_"+opt.name+"s"+str(opt.seed)+".csv", 'w') as resultFile:
                wr = csv.writer(resultFile, dialect='excel')
                wr.writerow(keys)
                wr.writerows(zip(*[final_grads[key] for key in keys]))
            keys = sorted(final_grads_m.keys())
            with open("./"+opt.resdir+"/smglite_"+opt.name+"s"+str(opt.seed)+".csv", 'w') as resultFile:
                wr = csv.writer(resultFile, dialect='excel')
                wr.writerow(keys)
                wr.writerows(zip(*[final_grads_m[key] for key in keys]))

            if opt.display_id >= 1:
                visualizer.plot_grads(final_grads)
                visualizer.plot_grads_m(final_grads_m)

        grads.clear()
        grads_m.clear()

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

def save_net(net,filename,resdir):
    save_filename = 'net_'+filename+'.pth'
    save_path = './'+resdir+'/'
    torch.save(net.state_dict(), save_path + save_filename)

def load_net(net,filename,resdir):
    save_filename = 'net_'+filename+'.pth'
    save_path = './'+resdir+'/'
    net.load_state_dict(torch.load(save_path + save_filename))

def save_grad_global(name):
    def hook(self, grad_input, grad_output):
        if "lstm_" in name:
            gradient_lstm = grad_output[1].norm().item()
            grads[name].append(gradient_lstm)
        else:
            gradient = grad_output[0].norm().item()
            grads[name].append(gradient)
    return hook


def save_grad(name):
    def hook(grad):
        if "lstm" in name:
            hidden_size = int(grad.shape[0]/4)
            grad_i = grad[0:hidden_size].data.norm().item()
            grad_i_m = grad[0:hidden_size].data.abs().mean().item()
            grads[name+"_i"].append(grad_i)
            grads_m[name+"_i"].append(grad_i_m)
            grad_f = grad[hidden_size:hidden_size*2].data.norm().item()
            grad_f_m = grad[hidden_size:hidden_size*2].data.abs().mean().item()
            grads[name + "_f"].append(grad_f)
            grads_m[name + "_f"].append(grad_f_m)
            grad_g = grad[hidden_size*2:hidden_size*3].data.norm().item()
            grad_g_m = grad[hidden_size*2:hidden_size*3].data.abs().mean().item()
            grads[name + "_g"].append(grad_g)
            grads_m[name + "_g"].append(grad_g_m)
            grad_o = grad[hidden_size*3:hidden_size*4].data.norm().item()
            grad_o_m = grad[hidden_size*3:hidden_size*4].data.abs().mean().item()
            grads[name + "_o"].append(grad_o)
            grads_m[name + "_o"].append(grad_o_m)

        else:
            grads[name].append(grad.data.norm().item())
            grads_m[name].append(grad.data.abs().mean().item())
    return hook



if __name__ == '__main__':
    main()