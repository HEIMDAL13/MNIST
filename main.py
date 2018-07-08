from __future__ import print_function
import argparse
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from options import Options
from model import Net, Net_LSTM, Net_one
from data import CustomDatasetDataLoader
from solver import Solver
import numpy as np
from visualizer import Visualizer


def main():
    opt = Options().parse()
    visualizer = Visualizer(opt)
    average(opt,visualizer)
    #train_test(opt,visualizer)

def train_test(opt,visualizer):
    torch.manual_seed(opt.seed)
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    data_loader = CustomDatasetDataLoader()
    data_loader.initialize(opt)

    visualizer.start_timmer()

    if opt.lstm:
        model = Net_LSTM(opt).to(opt.device)
        print("LSTM model created")
    elif opt.one_layer:
        model = Net_one(opt).to(opt.device)
        print("One layer model created")
    else:
        model = Net(opt).to(opt.device)
        print("Two layer model created")
    #best_model = copy.deepcopy(model)
    save_net(model,visualizer.filename)

    best_epoch = 0
    solver = Solver(model, data_loader, opt,visualizer)
    acc_past = 0
    #visualizer.set_filename(opt.name+str(opt.first_size)+"x"+str(opt.n_hidden*4)+'lstm'+str(opt.lstm)+'drop'+str(opt.drop1)+"-"+ str(opt.drop2)+"seed"+str(opt.seed) + "subs" + str(opt.train_size))
    visualizer.set_filename(opt.name+str(opt.first_size)+"x"+str(opt.n_hidden*4)+'lstm'+str(opt.lstm)+'drop'+str(int(opt.drop1*100))+"-"+ str(int(opt.drop2*100)) + "subs" + str(opt.train_size)+"1_lay"+ str(opt.one_layer)+"seed"+str(opt.seed))
    visualizer.write_options()
    visualizer.write_network_structure()
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    visualizer.write_text("Total Network parameters: " + str(pytorch_total_params))


    train_losses=[]
    val_losses=[]

    for epoch in range(1, opt.epochs + 1):
        train_losses.append(solver.train(epoch))
        val_acc,val_loss = solver.val()
        val_losses.append(val_loss)
        if acc_past<val_acc:
            acc_past=val_acc
            save_net(model,visualizer.filename)
            #best_model = copy.deepcopy(model)
            best_epoch = epoch

    visualizer.write_text("Evaluation last model:")
    acc_last, loss_last = solver.test()
    visualizer.write_text("Evaluation best model epoch " +str(best_epoch))

    load_net(model,visualizer.filename)
    acc_best, loss_best = solver.test()
    #acc_best, loss_best = solver.test(model=best_model)
    visualizer.write_time()
    visualizer.plot_trainval(train_losses,val_losses)
    visualizer.flush_to_file()

    return train_losses, val_losses, acc_best, acc_last, loss_best, loss_last, best_epoch

def average(opt,visualizer):
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
        train_losses, val_losses, acc_best, acc_last, loss_best, loss_last, best_epoch = train_test(opt,visualizer)
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

    visualizer.set_filename("AVERAGE"+opt.name+str(opt.first_size)+"x"+str(opt.n_hidden*4)+'lstm'+str(opt.lstm)+'drop'+str(int(opt.drop1*100))+"-"+ str(int(opt.drop2*100)) + "subs" + str(opt.train_size)+"1_lay"+ str(opt.one_layer))
    visualizer.write_options()
    #visualizer.write_num_parameters()
    visualizer.write_network_structure()
    visualizer.plot_trainval(train_losses_av, val_losses_av)
    visualizer.write_text("\nBest Model: acc: {:.4f} +- {:.4f}".format(acc_best_m, acc_best_std))
    visualizer.write_text("Last Model: acc: {:.4f} +- {:.4f}".format(acc_last_m, acc_last_std))
    visualizer.write_text("Best epoch: {:.2f}±{:.2f} \n".format(best_epoch_m, best_epoch_std))
    visualizer.write_text("Best Model: loss: {:.4f} +- {:.4f}".format(loss_best_m, loss_best_std))
    visualizer.write_text("Last Model: loss: {:.4f} +- {:.4f}".format(loss_last_m, loss_last_std))
    visualizer.flush_to_file()


def save_net(net,filename):
    save_filename = filename+'.pth'
    save_path = './results/plot'
    torch.save(net.state_dict(), save_path + save_filename)

def load_net(net,filename):
    save_filename = filename+'.pth'
    save_path = './results/plot'
    net.load_state_dict(torch.load(save_path + save_filename))


if __name__ == '__main__':
    main()