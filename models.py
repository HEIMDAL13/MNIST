import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from LSTMCell_custom import LSTMCell_custom
class Net_1FC(nn.Module):
    def __init__(self, opt):
        super(Net_1FC, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(opt.input_size, opt.first_size), nn.ReLU(), nn.Dropout(p=opt.drop1))
        self.fc3 = nn.Linear(opt.first_size, 10)
        self.opt = opt

    def forward(self, x):
        input_inter = x.view(x.size(0), -1)
        x = self.fc1(input_inter)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class Net_2FC(nn.Module):
    def __init__(self,opt):

        super(Net_2FC, self).__init__()
        self.opt = opt
        self.fc1 = nn.Sequential(nn.Linear(opt.input_size, opt.first_size),nn.ReLU(),nn.Dropout(p=opt.drop1))
        self.fc2 = nn.Sequential(nn.Linear(opt.first_size, opt.second_size),nn.ReLU(),nn.Dropout(p=opt.drop2))
        if self.opt.batch_norm1 == 1:
            self.batch1 = nn.BatchNorm1d(opt.first_size)
        if self.opt.batch_norm2 == 1:
            self.batch2 = nn.BatchNorm1d(opt.second_size)
        self.fc3 = nn.Linear(opt.second_size, 10)
    def forward(self, x):
        input_inter = x.view(x.size(0), -1)
        x = self.fc1(input_inter)
        if self.opt.batch_norm1 == 1:
            x = self.batch1(x)
        x = self.fc2(x)
        if self.opt.batch_norm2 == 1:
            x = self.batch2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class Net_3FC(nn.Module):
    def __init__(self,opt):
        super(Net_3FC, self).__init__()
        self.opt = opt
        self.fc1 = nn.Sequential(nn.Linear(opt.input_size, opt.first_size),nn.ReLU(),nn.Dropout(p=opt.drop1))
        self.fc2 = nn.Sequential(nn.Linear(opt.first_size, opt.second_size),nn.ReLU(),nn.Dropout(p=opt.drop2))
        self.fc3 = nn.Sequential(nn.Linear(opt.second_size, opt.third_size),nn.ReLU(),nn.Dropout(p=opt.drop3))
        if self.opt.batch_norm1 == 1:
            self.batch1 = nn.BatchNorm1d(opt.first_size)
        if self.opt.batch_norm2 == 1:
            self.batch2 = nn.BatchNorm1d(opt.second_size)
        if self.opt.batch_norm3 == 1:
            self.batch3 = nn.BatchNorm1d(opt.third_size)
        self.fc4 = nn.Linear(opt.third_size, 10)

    def forward(self, x):
        input_inter = x.view(x.size(0), -1)
        x = self.fc1(input_inter)
        if self.opt.batch_norm1 == 1:
            x = self.batch1(x)
        x = self.fc2(x)
        if self.opt.batch_norm2 == 1:
            x = self.batch2(x)
        x = self.fc3(x)
        if self.opt.batch_norm3 == 1:
            x = self.batch3(x)
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


class Net_RNN(nn.Module):
    def __init__(self,opt):
        super(Net_RNN, self).__init__()

        self.n_hidden = opt.n_hidden
        self.fc1 = nn.Sequential(nn.Linear(opt.input_size, opt.first_size), nn.ReLU(), nn.Dropout(p=opt.drop1))
        self.dropout_lstm = nn.Dropout(p=opt.drop2)
        self.fc3 = nn.Linear(opt.n_hidden * 4, 10)
        self.opt = opt
        self.rnn_pose_lr = nn.RNN(input_size=opt.shape_lstm, hidden_size=self.n_hidden, bidirectional=True)
        self.rnn_pose_ud = nn.RNN(input_size=int(opt.first_size/opt.shape_lstm), hidden_size=self.n_hidden, bidirectional=True)

    def init_hidden_(self, batch_size, device):
        '''Return initialized hidden states and cell states for each biodirectional lstm cell'''
        return torch.randn(2, batch_size, self.n_hidden).to(device)


    def forward(self, x):
        batch_size = x.size()[0]
        input_inter = x.view(x.size(0), -1)
        x = self.fc1(input_inter)
        hidden_rightleft = self.init_hidden_(batch_size, x.device)
        hidden_downup = self.init_hidden_(batch_size, x.device)

        input_lstm = x.view(x.size(0), self.opt.shape_lstm, -1)
        x_rightleft = input_lstm.permute(1, 0, 2)
        x_downup = input_lstm.permute(2, 0, 1)
        outputlr, hidden_state_lr = self.rnn_pose_lr(x_rightleft, hidden_rightleft)
        outputud, hidden_state_ud = self.rnn_pose_ud(x_downup, hidden_downup)
        final_output_lstm = torch.cat(
            (
            hidden_state_lr[0, :, :], hidden_state_lr[1, :, :], hidden_state_ud[0, :, :], hidden_state_ud[1, :, :]),
            1)
        x = self.dropout_lstm(final_output_lstm)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class Net_GRU(nn.Module):
    def __init__(self,opt):
        super(Net_GRU, self).__init__()

        self.n_hidden = opt.n_hidden
        self.fc1 = nn.Sequential(nn.Linear(opt.input_size, opt.first_size), nn.ReLU(), nn.Dropout(p=opt.drop1))
        self.dropout_lstm = nn.Dropout(p=opt.drop2)
        self.fc3 = nn.Linear(opt.n_hidden * 4, 10)
        self.opt = opt
        self.gru_pose_lr = nn.GRU(input_size=opt.shape_lstm, hidden_size=self.n_hidden, bidirectional=True)
        self.gru_pose_ud = nn.GRU(input_size=int(opt.first_size/opt.shape_lstm), hidden_size=self.n_hidden, bidirectional=True)

    def init_hidden_(self, batch_size, device):
        '''Return initialized hidden states and cell states for each biodirectional lstm cell'''
        return torch.randn(2, batch_size, self.n_hidden).to(device)


    def forward(self, x):
        batch_size = x.size()[0]
        input_inter = x.view(x.size(0), -1)
        x = self.fc1(input_inter)
        hidden_rightleft = self.init_hidden_(batch_size, x.device)
        hidden_downup = self.init_hidden_(batch_size, x.device)

        input_lstm = x.view(x.size(0), self.opt.shape_lstm, -1)
        x_rightleft = input_lstm.permute(1, 0, 2)
        x_downup = input_lstm.permute(2, 0, 1)
        outputlr, hidden_state_lr = self.gru_pose_lr(x_rightleft, hidden_rightleft)
        outputud, hidden_state_ud = self.gru_pose_ud(x_downup, hidden_downup)
        final_output_lstm = torch.cat(
            (
            hidden_state_lr[0, :, :], hidden_state_lr[1, :, :], hidden_state_ud[0, :, :], hidden_state_ud[1, :, :]),
            1)
        x = self.dropout_lstm(final_output_lstm)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class Net_LSTM(nn.Module):
    def __init__(self,opt):
        super(Net_LSTM, self).__init__()
        self.n_hidden = opt.n_hidden
        self.fc1 = nn.Sequential(nn.Linear(opt.input_size, opt.first_size),nn.ReLU(),nn.Dropout(p=opt.drop1))
        self.dropout_lstm = nn.Dropout(p=opt.drop2)
        self.fc3 = nn.Linear(opt.n_hidden*4, 10)
        self.opt = opt
        self.lstm_pose_lr = nn.LSTM(input_size=int(opt.first_size/opt.shape_lstm), hidden_size=self.n_hidden, bidirectional=True)
        self.lstm_pose_ud = nn.LSTM(input_size=opt.shape_lstm, hidden_size=self.n_hidden, bidirectional=True)

    def init_hidden_(self, batch_size, device):
        '''Return initialized hidden states and cell states for each biodirectional lstm cell'''
        return (torch.randn(2, batch_size, self.n_hidden).to(device),
                torch.randn(2, batch_size, self.n_hidden).to(device))


    def forward(self, x):
        batch_size = x.size()[0]
        input_inter = x.view(x.size(0), -1)
        x = self.fc1(input_inter)
        hidden_rightleft = self.init_hidden_(batch_size, x.device)
        hidden_downup = self.init_hidden_(batch_size, x.device)
        input_lstm = x.view(x.size(0), self.opt.shape_lstm, -1)
        x_rightleft = input_lstm.permute(1, 0, 2)
        x_downup = input_lstm.permute(2, 0, 1)
        outputlr, (hidden_state_lr, cell_state_lr) = self.lstm_pose_lr(x_rightleft, hidden_rightleft)
        outputud, (hidden_state_ud, cell_state_ud) = self.lstm_pose_ud(x_downup, hidden_downup)
        final_output_lstm = torch.cat(
        (hidden_state_lr[0, :, :], hidden_state_lr[1, :, :], hidden_state_ud[0, :, :], hidden_state_ud[1, :, :]), 1)
        x = self.dropout_lstm(final_output_lstm)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class Net_LSTM_cell(nn.Module):
    def __init__(self,opt):
        super(Net_LSTM_cell, self).__init__()
        self.n_hidden = opt.n_hidden
        self.fc1 = nn.Sequential(nn.Linear(opt.input_size, opt.first_size),nn.ReLU(),nn.Dropout(p=opt.drop1))
        self.dropout_lstm = nn.Dropout(p=opt.drop2)
        self.fc3 = nn.Linear(opt.n_hidden*4, 10)
        self.opt = opt
        self.lstm_pose_l = LSTMCell_custom(input_size=int(opt.first_size/opt.shape_lstm), hidden_size=self.n_hidden, print=True)
        self.lstm_pose_r = LSTMCell_custom(input_size=int(opt.first_size/opt.shape_lstm), hidden_size=self.n_hidden)
        self.lstm_pose_u = LSTMCell_custom(input_size=opt.shape_lstm, hidden_size=self.n_hidden)
        self.lstm_pose_d = LSTMCell_custom(input_size=opt.shape_lstm, hidden_size=self.n_hidden)

    def init_hidden_(self, batch_size, device):
        '''Return initialized hidden states and cell states for each biodirectional lstm cell'''
        return (torch.randn(batch_size, self.n_hidden).to(device),torch.randn(batch_size, self.n_hidden).to(device))

    def flip(self, x, dim):
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                    dtype=torch.long, device=x.device)
        return x[tuple(indices)]

    def forward(self, x):
        batch_size = x.size()[0]
        input_inter = x.view(x.size(0), -1)
        x = self.fc1(input_inter)
        hidden_left = self.init_hidden_(batch_size, x.device)
        hidden_right = self.init_hidden_(batch_size, x.device)
        hidden_up = self.init_hidden_(batch_size, x.device)
        hidden_down = self.init_hidden_(batch_size, x.device)

        input_lstm = x.view(x.size(0), self.opt.shape_lstm, -1)
        x_left = input_lstm.permute(1, 0, 2)
        x_right = self.flip(x_left,0)
        x_up = input_lstm.permute(2, 0, 1)
        x_down = self.flip(x_up,0)
        print("NEW ITERATION!")
        for i in range(self.opt.shape_lstm):
            hidden_left = self.lstm_pose_l(x_left[i,:,:], hidden_left)
            hidden_right = self.lstm_pose_r(x_right[i,:,:], hidden_right)
            hidden_up = self.lstm_pose_d(x_up[i,:,:], hidden_up)
            hidden_down = self.lstm_pose_d(x_down[i,:,:], hidden_down)

        final_output_lstm = torch.cat((hidden_left[0], hidden_right[0], hidden_up[0], hidden_down[0]), 1)
        x = self.dropout_lstm(final_output_lstm)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class Net_LSTM2(nn.Module):
    def __init__(self, opt):
        super(Net_LSTM2, self).__init__()
        self.n_hidden = opt.n_hidden
        self.fc1 = nn.Sequential(nn.Linear(opt.input_size, opt.first_size), nn.ReLU(), nn.Dropout(p=opt.drop1))
        self.fc2 = nn.Sequential(nn.Linear(opt.first_size, opt.second_size), nn.ReLU(), nn.Dropout(p=opt.drop2))
        self.dropout_lstm = nn.Dropout(p=opt.drop3)
        self.fc3 = nn.Linear(opt.n_hidden * 4, 10)
        self.opt = opt
        if opt.shape_lstm == 0:
            self.n_hidden = self.n_hidden * 2
            self.lstm_pose_lr = nn.LSTM(input_size=1, hidden_size=self.n_hidden, bidirectional=True)
        else:
            self.lstm_pose_lr = nn.LSTM(input_size=int(opt.second_size / opt.shape_lstm), hidden_size=self.n_hidden,
                                        bidirectional=True, num_layers=opt.lstm_layers)
            self.lstm_pose_ud = nn.LSTM(input_size=opt.shape_lstm, hidden_size=self.n_hidden, bidirectional=True, num_layers=opt.lstm_layers)

    def init_hidden_(self, batch_size, device):
        '''Return initialized hidden states and cell states for each biodirectional lstm cell'''
        return (torch.randn(self.opt.lstm_layers*2, batch_size, self.n_hidden).to(device),
                torch.randn(self.opt.lstm_layers*2, batch_size, self.n_hidden).to(device))


    def forward(self, x):
        batch_size = x.size()[0]
        input_inter = x.view(x.size(0), -1)
        x = self.fc1(input_inter)
        x = self.fc2(x)
        hidden_rightleft = self.init_hidden_(batch_size, x.device)
        hidden_downup = self.init_hidden_(batch_size, x.device)


        if self.opt.shape_lstm == 0:
            input_lstm = x.view(x.size(0), 1, -1)
            x_rightleft =input_lstm.permute(2, 0, 1)
            outputlr, (hidden_state_lr, cell_state_lr) = self.lstm_pose_lr(x_rightleft,hidden_rightleft)
            final_output_lstm = torch.cat((hidden_state_lr[0, :, :], hidden_state_lr[1, :, :]), 1)
        else:
            input_lstm = x.view(x.size(0), self.opt.shape_lstm, -1)
            x_rightleft = input_lstm.permute(1, 0, 2)
            x_downup = input_lstm.permute(2, 0, 1)
            outputlr, (hidden_state_lr, cell_state_lr) = self.lstm_pose_lr(x_rightleft, hidden_rightleft)
            outputud, (hidden_state_ud, cell_state_ud) = self.lstm_pose_ud(x_downup, hidden_downup)
            final_output_lstm = torch.cat(
            (hidden_state_lr[0, :, :], hidden_state_lr[1, :, :], hidden_state_ud[0, :, :], hidden_state_ud[1, :, :]), 1)
        x = self.dropout_lstm(final_output_lstm)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class Net_LSTM25(nn.Module):
    def __init__(self, opt):
        super(Net_LSTM25, self).__init__()
        self.n_hidden = opt.n_hidden
        self.fc1 = nn.Sequential(nn.Linear(opt.input_size, opt.first_size), nn.ReLU(), nn.Dropout(p=opt.drop1))
        self.fc2 = nn.Sequential(nn.Linear(opt.n_hidden * 4, 128), nn.ReLU(), nn.Dropout(p=opt.drop2))
        self.dropout_lstm = nn.Dropout(p=opt.drop3)
        self.fc3 = nn.Linear(128, 10)
        self.opt = opt

        self.lstm_pose_lr = nn.LSTM(input_size=int(opt.second_size / opt.shape_lstm), hidden_size=self.n_hidden,
                                    bidirectional=True, num_layers=opt.lstm_layers)
        self.lstm_pose_ud = nn.LSTM(input_size=opt.shape_lstm, hidden_size=self.n_hidden, bidirectional=True, num_layers=opt.lstm_layers)

    def init_hidden_(self, batch_size, device):
        '''Return initialized hidden states and cell states for each biodirectional lstm cell'''
        return (torch.randn(self.opt.lstm_layers*2, batch_size, self.n_hidden).to(device),
                torch.randn(self.opt.lstm_layers*2, batch_size, self.n_hidden).to(device))


    def forward(self, x):
        batch_size = x.size()[0]
        input_inter = x.view(x.size(0), -1)
        x = self.fc1(input_inter)
        hidden_rightleft = self.init_hidden_(batch_size, x.device)
        hidden_downup = self.init_hidden_(batch_size, x.device)


        if self.opt.shape_lstm == 0:
            input_lstm = x.view(x.size(0), 1, -1)
            x_rightleft =input_lstm.permute(2, 0, 1)
            outputlr, (hidden_state_lr, cell_state_lr) = self.lstm_pose_lr(x_rightleft,hidden_rightleft)
            final_output_lstm = torch.cat((hidden_state_lr[0, :, :], hidden_state_lr[1, :, :]), 1)
        else:
            input_lstm = x.view(x.size(0), self.opt.shape_lstm, -1)
            x_rightleft = input_lstm.permute(1, 0, 2)
            x_downup = input_lstm.permute(2, 0, 1)
            outputlr, (hidden_state_lr, cell_state_lr) = self.lstm_pose_lr(x_rightleft, hidden_rightleft)
            outputud, (hidden_state_ud, cell_state_ud) = self.lstm_pose_ud(x_downup, hidden_downup)
            final_output_lstm = torch.cat(
            (hidden_state_lr[0, :, :], hidden_state_lr[1, :, :], hidden_state_ud[0, :, :], hidden_state_ud[1, :, :]), 1)
        x = self.dropout_lstm(final_output_lstm)
        x = self.fc2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)




class LeNet(nn.Module):
    def __init__(self,opt):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(p=opt.drop1)
        self.fc1 = nn.Sequential(nn.Linear(1000, opt.first_size), nn.ReLU(), nn.Dropout(p=opt.drop2))
        self.fc2 = nn.Sequential(nn.Linear(opt.first_size, opt.second_size), nn.ReLU(), nn.Dropout(p=opt.drop3))
        self.fc3 = nn.Linear(opt.second_size, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 1000)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class LeNet_LSTM(nn.Module):
    def __init__(self,opt):
        super(LeNet_LSTM, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(p=opt.drop1)
        self.fc1 = nn.Sequential(nn.Linear(1000, opt.first_size), nn.ReLU(), nn.Dropout(p=opt.drop2))
        self.fc3 = nn.Linear(opt.n_hidden*4, 10)
        self.opt = opt
        self.n_hidden = opt.n_hidden
        self.dropout_lstm = nn.Dropout(p=opt.drop3)


        self.lstm_pose_lr = nn.LSTM(input_size=int(opt.first_size / opt.shape_lstm), hidden_size=self.n_hidden, bidirectional=True, num_layers=opt.lstm_layers)
        self.lstm_pose_ud = nn.LSTM(input_size=opt.shape_lstm, hidden_size=self.n_hidden, bidirectional=True, num_layers=opt.lstm_layers)

    def init_hidden_(self, batch_size, device):
        '''Return initialized hidden states and cell states for each biodirectional lstm cell'''
        return (torch.randn(self.opt.lstm_layers*2, batch_size, self.n_hidden).to(device),
                torch.randn(self.opt.lstm_layers*2, batch_size, self.n_hidden).to(device))

    def forward(self, x):

        batch_size = x.size()[0]

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 1000)
        x = self.fc1(x)
        hidden_rightleft = self.init_hidden_(batch_size, x.device)
        hidden_downup = self.init_hidden_(batch_size, x.device)

        input_lstm = x.view(x.size(0), self.opt.shape_lstm, -1)
        x_rightleft = input_lstm.permute(1, 0, 2)
        x_downup = input_lstm.permute(2, 0, 1)
        outputlr, (hidden_state_lr, cell_state_lr) = self.lstm_pose_lr(x_rightleft, hidden_rightleft)
        outputud, (hidden_state_ud, cell_state_ud) = self.lstm_pose_ud(x_downup, hidden_downup)
        final_output_lstm = torch.cat(
            (
            hidden_state_lr[0, :, :], hidden_state_lr[1, :, :], hidden_state_ud[0, :, :], hidden_state_ud[1, :, :]),
            1)
        x = self.dropout_lstm(final_output_lstm)
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)



class LeNet_LSTM_first(nn.Module):
    def __init__(self,opt):
        super(LeNet_LSTM_first, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(p=opt.drop1)
        self.conv3 = nn.Conv2d(40, 1, kernel_size=1)
        self.fc1 = nn.Sequential(nn.Linear(1000, opt.first_size), nn.ReLU(), nn.Dropout(p=opt.drop2))
        self.fc3 = nn.Linear(opt.n_hidden*4, 10)
        self.opt = opt
        self.n_hidden = opt.n_hidden
        self.dropout_lstm = nn.Dropout(p=opt.drop3)


        self.lstm_pose_lr = nn.LSTM(input_size=22, hidden_size=self.n_hidden, bidirectional=True, num_layers=opt.lstm_layers)
        self.lstm_pose_ud = nn.LSTM(input_size=22, hidden_size=self.n_hidden, bidirectional=True, num_layers=opt.lstm_layers)

    def init_hidden_(self, batch_size, device):
        '''Return initialized hidden states and cell states for each biodirectional lstm cell'''
        return (torch.randn(self.opt.lstm_layers*2, batch_size, self.n_hidden).to(device),
                torch.randn(self.opt.lstm_layers*2, batch_size, self.n_hidden).to(device))

    def forward(self, x):

        batch_size = x.size()[0]

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2_drop(self.conv2(x)))
        x = F.relu(self.conv3(x))
        input_lstm = np.squeeze(x)
        hidden_rightleft = self.init_hidden_(batch_size, x.device)
        hidden_downup = self.init_hidden_(batch_size, x.device)
        x_rightleft = input_lstm.permute(1, 0, 2)
        x_downup = input_lstm.permute(2, 0, 1)
        outputlr, (hidden_state_lr, cell_state_lr) = self.lstm_pose_lr(x_rightleft, hidden_rightleft)
        outputud, (hidden_state_ud, cell_state_ud) = self.lstm_pose_ud(x_downup, hidden_downup)
        final_output_lstm = torch.cat(
            (
            hidden_state_lr[0, :, :], hidden_state_lr[1, :, :], hidden_state_ud[0, :, :], hidden_state_ud[1, :, :]),
            1)
        x = self.dropout_lstm(final_output_lstm)
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)

class LeNet_plus(nn.Module):
    def __init__(self, opt):
        super(LeNet_plus, self).__init__()
        self.opt = opt
        self.conv1 = nn.Conv2d(3, 20, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 40, 5)
        self.fc1 = nn.Linear(40 * 5 * 5, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 10)
        self.fc_drop = nn.Dropout(p=opt.drop2)
        if self.opt.batch_norm1 == 1:
            self.batch1 = nn.BatchNorm1d(opt.second_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 40 * 5 * 5)
        x = F.relu(self.fc1(x))
        if self.opt.batch_norm1 == 1:
            x = self.batch1(x)
        x = F.relu(self.fc2(x))
        x = self.fc_drop(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)



class LeNet_plus_LSTM(nn.Module):
    def __init__(self, opt):
        super(LeNet_plus_LSTM, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 40, 5)
        self.fc1 = nn.Linear(40 * 5 * 5, opt.first_size)
        self.fc3 = nn.Linear(opt.n_hidden*4, 10)
        self.fc_drop = nn.Dropout(p=opt.drop2)
        self.opt = opt
        self.n_hidden = opt.n_hidden
        self.dropout_lstm = nn.Dropout(p=opt.drop3)

        self.lstm_pose_lr = nn.LSTM(input_size=int(opt.first_size / opt.shape_lstm), hidden_size=self.n_hidden,
                                    bidirectional=True)
        self.lstm_pose_ud = nn.LSTM(input_size=opt.shape_lstm, hidden_size=self.n_hidden, bidirectional=True)

    def init_hidden_(self, batch_size, device):
        '''Return initialized hidden states and cell states for each biodirectional lstm cell'''
        return (torch.randn(2, batch_size, self.n_hidden).to(device),
                torch.randn(2, batch_size, self.n_hidden).to(device))



    def forward(self, x):

        batch_size = x.size()[0]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 40 * 5 * 5)
        x = F.relu(self.fc1(x))
        hidden_rightleft = self.init_hidden_(batch_size, x.device)
        hidden_downup = self.init_hidden_(batch_size, x.device)

        input_lstm = x.view(x.size(0), self.opt.shape_lstm, -1)
        x_rightleft = input_lstm.permute(1, 0, 2)
        x_downup = input_lstm.permute(2, 0, 1)
        outputlr, (hidden_state_lr, cell_state_lr) = self.lstm_pose_lr(x_rightleft, hidden_rightleft)
        outputud, (hidden_state_ud, cell_state_ud) = self.lstm_pose_ud(x_downup, hidden_downup)
        final_output_lstm = torch.cat(
            (
            hidden_state_lr[0, :, :], hidden_state_lr[1, :, :], hidden_state_ud[0, :, :], hidden_state_ud[1, :, :]),
            1)
        x = self.dropout_lstm(final_output_lstm)
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)



class LeNet_plus_RNN(nn.Module):
    def __init__(self, opt):
        super(LeNet_plus_RNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 40, 5)
        self.fc1 = nn.Linear(40 * 5 * 5, opt.first_size)
        self.fc3 = nn.Linear(opt.n_hidden*4, 10)
        self.fc_drop = nn.Dropout(p=opt.drop2)
        self.opt = opt
        self.n_hidden = opt.n_hidden
        self.dropout_lstm = nn.Dropout(p=opt.drop3)

        self.rnn_pose_lr = nn.RNN(input_size=opt.shape_lstm, hidden_size=self.n_hidden, bidirectional=True)
        self.rnn_pose_ud = nn.RNN(input_size=int(opt.first_size / opt.shape_lstm), hidden_size=self.n_hidden,
                                  bidirectional=True)

    def init_hidden_(self, batch_size, device):
        '''Return initialized hidden states and cell states for each biodirectional lstm cell'''
        return torch.randn(2, batch_size, self.n_hidden).to(device)



    def forward(self, x):

        batch_size = x.size()[0]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 40 * 5 * 5)
        x = F.relu(self.fc1(x))
        hidden_rightleft = self.init_hidden_(batch_size, x.device)
        hidden_downup = self.init_hidden_(batch_size, x.device)

        input_lstm = x.view(x.size(0), self.opt.shape_lstm, -1)
        x_rightleft = input_lstm.permute(1, 0, 2)
        x_downup = input_lstm.permute(2, 0, 1)
        outputlr, hidden_state_lr = self.rnn_pose_lr(x_rightleft, hidden_rightleft)
        outputud, hidden_state_ud = self.rnn_pose_ud(x_downup, hidden_downup)
        final_output_lstm = torch.cat(
            (
            hidden_state_lr[0, :, :], hidden_state_lr[1, :, :], hidden_state_ud[0, :, :], hidden_state_ud[1, :, :]),
            1)
        x = self.dropout_lstm(final_output_lstm)
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)



class LeNet_RNN(nn.Module):
    def __init__(self,opt):
        super(LeNet_RNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(p=opt.drop1)
        self.fc1 = nn.Sequential(nn.Linear(1000, opt.first_size), nn.ReLU(), nn.Dropout(p=opt.drop2))
        self.fc3 = nn.Linear(opt.n_hidden*4, 10)
        self.opt = opt
        self.n_hidden = opt.n_hidden
        self.dropout_lstm = nn.Dropout(p=opt.drop3)

        self.rnn_pose_lr = nn.RNN(input_size=opt.shape_lstm, hidden_size=self.n_hidden, bidirectional=True)
        self.rnn_pose_ud = nn.RNN(input_size=int(opt.first_size / opt.shape_lstm), hidden_size=self.n_hidden,
                                  bidirectional=True)

    def init_hidden_(self, batch_size, device):
        '''Return initialized hidden states and cell states for each biodirectional lstm cell'''
        return torch.randn(2, batch_size, self.n_hidden).to(device)


    def forward(self, x):

        batch_size = x.size()[0]

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 1000)
        x = self.fc1(x)
        hidden_rightleft = self.init_hidden_(batch_size, x.device)
        hidden_downup = self.init_hidden_(batch_size, x.device)


        input_lstm = x.view(x.size(0), self.opt.shape_lstm, -1)
        x_rightleft = input_lstm.permute(1, 0, 2)
        x_downup = input_lstm.permute(2, 0, 1)
        outputlr, hidden_state_lr = self.rnn_pose_lr(x_rightleft, hidden_rightleft)
        outputud, hidden_state_ud = self.rnn_pose_ud(x_downup, hidden_downup)
        final_output_lstm = torch.cat(
            (
            hidden_state_lr[0, :, :], hidden_state_lr[1, :, :], hidden_state_ud[0, :, :], hidden_state_ud[1, :, :]),
            1)
        x = self.dropout_lstm(final_output_lstm)
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)


class LeNet_GRU(nn.Module):
    def __init__(self,opt):
        super(LeNet_GRU, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(p=opt.drop1)
        self.fc1 = nn.Sequential(nn.Linear(1000, opt.first_size), nn.ReLU(), nn.Dropout(p=opt.drop2))
        self.fc3 = nn.Linear(opt.n_hidden*4, 10)
        self.opt = opt
        self.n_hidden = opt.n_hidden
        self.dropout_lstm = nn.Dropout(p=opt.drop3)

        self.gru_pose_lr = nn.GRU(input_size=opt.shape_lstm, hidden_size=self.n_hidden, bidirectional=True)
        self.gru_pose_ud = nn.GRU(input_size=int(opt.first_size / opt.shape_lstm), hidden_size=self.n_hidden,
                                  bidirectional=True)

    def init_hidden_(self, batch_size, device):
        '''Return initialized hidden states and cell states for each biodirectional lstm cell'''
        return torch.randn(2, batch_size, self.n_hidden).to(device)


    def forward(self, x):

        batch_size = x.size()[0]

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 1000)
        x = self.fc1(x)
        hidden_rightleft = self.init_hidden_(batch_size, x.device)
        hidden_downup = self.init_hidden_(batch_size, x.device)


        input_lstm = x.view(x.size(0), self.opt.shape_lstm, -1)
        x_rightleft = input_lstm.permute(1, 0, 2)
        x_downup = input_lstm.permute(2, 0, 1)
        outputlr, hidden_state_lr = self.gru_pose_lr(x_rightleft, hidden_rightleft)
        outputud, hidden_state_ud = self.gru_pose_ud(x_downup, hidden_downup)
        final_output_lstm = torch.cat(
            (
            hidden_state_lr[0, :, :], hidden_state_lr[1, :, :], hidden_state_ud[0, :, :], hidden_state_ud[1, :, :]),
            1)
        x = self.dropout_lstm(final_output_lstm)
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)


class LeNet_LSTM_exp(nn.Module):
    def __init__(self,opt):
        super(LeNet_LSTM_exp, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(p=opt.drop1)
        self.fc1 = nn.Sequential(nn.Linear(1000, opt.first_size), nn.ReLU(), nn.Dropout(p=opt.drop2))
        self.opt = opt
        self.n_hidden = opt.n_hidden
        self.dropout_lstm = nn.Dropout(p=opt.drop3)



        self.lstm_pose_lr = nn.LSTM(input_size=int(opt.first_size / opt.shape_lstm), hidden_size=self.n_hidden,
                                    bidirectional=True)
        self.lstm_pose_ud = nn.LSTM(input_size=opt.shape_lstm, hidden_size=self.n_hidden, bidirectional=True)

    def init_hidden_(self, batch_size, device):
        '''Return initialized hidden states and cell states for each biodirectional lstm cell'''
        return (torch.randn(2, batch_size, self.n_hidden).to(device),
                torch.randn(2, batch_size, self.n_hidden).to(device))

    def forward(self, x):

        batch_size = x.size()[0]

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 1000)
        x = self.fc1(x)
        hidden_rightleft = self.init_hidden_(batch_size, x.device)
        hidden_downup = self.init_hidden_(batch_size, x.device)

        input_lstm = x.view(x.size(0), self.opt.shape_lstm, -1)
        x_rightleft = input_lstm.permute(1, 0, 2)
        x_downup = input_lstm.permute(2, 0, 1)
        outputlr, (hidden_state_lr, cell_state_lr) = self.lstm_pose_lr(x_rightleft, hidden_rightleft)
        outputud, (hidden_state_ud, cell_state_ud) = self.lstm_pose_ud(x_downup, hidden_downup)
        final_output_lstm = torch.cat(
            (
            hidden_state_lr[0, :, :], hidden_state_lr[1, :, :], hidden_state_ud[0, :, :], hidden_state_ud[1, :, :]),
            1)
        x = self.dropout_lstm(final_output_lstm)

        return F.log_softmax(x, dim=1)



class LeNet_LSTM_mo(nn.Module):
    def __init__(self,opt):
        super(LeNet_LSTM_mo, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(p=opt.drop1)
        self.fc1 = nn.Sequential(nn.Linear(1000, opt.first_size), nn.ReLU(), nn.Dropout(p=opt.drop2))
        self.fc3 = nn.Linear(opt.third_size, 10)
        self.opt = opt
        self.n_hidden = opt.n_hidden
        self.dropout_lstm = nn.Dropout(p=opt.drop3)


        self.lstm_pose_lr = nn.LSTM(input_size=int(opt.first_size / opt.shape_lstm), hidden_size=self.n_hidden,
                                    bidirectional=True)
        self.lstm_pose_ud = nn.LSTM(input_size=opt.shape_lstm, hidden_size=self.n_hidden, bidirectional=True)

    def init_hidden_(self, batch_size, device):
        '''Return initialized hidden states and cell states for each biodirectional lstm cell'''
        return (torch.randn(2, batch_size, self.n_hidden).to(device),
                torch.randn(2, batch_size, self.n_hidden).to(device))

    def forward(self, x):

        batch_size = x.size()[0]

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 1000)
        x = self.fc1(x)
        hidden_rightleft = self.init_hidden_(batch_size, x.device)
        hidden_downup = self.init_hidden_(batch_size, x.device)

        input_lstm = x.view(x.size(0), self.opt.shape_lstm, -1)
        x_rightleft = input_lstm.permute(1, 0, 2)
        x_downup = input_lstm.permute(2, 0, 1)
        outputlr, (hidden_state_lr, cell_state_lr) = self.lstm_pose_lr(x_rightleft, hidden_rightleft)
        outputud, (hidden_state_ud, cell_state_ud) = self.lstm_pose_ud(x_downup, hidden_downup)

        #hhn_copylr = hidden_state_lr
        #hhn_copyud = hidden_state_ud
        if self.opt.full_output:
            hidden_state_lr = outputlr.transpose(0,1)
            hidden_state_ud = outputud.transpose(0,1)
        else:
            hidden_state_lr = hidden_state_lr.transpose(0, 1)
            hidden_state_ud = hidden_state_ud.transpose(0, 1)
        hidden_state_lr = hidden_state_lr.reshape(batch_size, -1)
        hidden_state_ud = hidden_state_ud.reshape(batch_size, -1)


        final_output_lstm = torch.cat((hidden_state_lr, hidden_state_ud),1)
        #final_output_lstm_copy = torch.cat(( hhn_copylr[0, :, :],  hhn_copylr[1, :, :], hhn_copyud[0, :, :], hhn_copyud[1, :, :]),1)
        x = self.dropout_lstm(final_output_lstm)
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)


class LeNet_LSTM_directconv(nn.Module):
    def __init__(self,opt):
        super(LeNet_LSTM_directconv, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(p=opt.drop1)
        self.conv3 = nn.Conv2d(40, 1, kernel_size=1)
        self.fc1 = nn.Sequential(nn.Linear(1000, opt.first_size), nn.ReLU(), nn.Dropout(p=opt.drop2))
        self.fc3 = nn.Linear(opt.n_hidden*4, 10)
        self.opt = opt
        self.n_hidden = opt.n_hidden
        self.dropout_lstm = nn.Dropout(p=opt.drop3)

        self.lstm_pose_lr = nn.LSTM(input_size=int(opt.first_size / opt.shape_lstm), hidden_size=self.n_hidden,
                                    bidirectional=True, num_layers=opt.lstm_layers)
        self.lstm_pose_ud = nn.LSTM(input_size=opt.shape_lstm, hidden_size=self.n_hidden, bidirectional=True, num_layers=opt.lstm_layers)

        if opt.full_output:
            self.lstm_pose_lr2 = nn.LSTM(input_size=int(opt.first_size / opt.shape_lstm), hidden_size=self.n_hidden,
                                        bidirectional=True)
            self.lstm_pose_ud2 = nn.LSTM(input_size=opt.shape_lstm, hidden_size=self.n_hidden, bidirectional=True)

    def init_hidden_(self, batch_size, device):
        '''Return initialized hidden states and cell states for each biodirectional lstm cell'''
        return (torch.randn(2, batch_size, self.n_hidden).to(device),
                torch.randn(2, batch_size, self.n_hidden).to(device))

    def forward(self, x):

        batch_size = x.size()[0]

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(self.conv2_drop(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 1000)
        x = self.fc1(x)
        hidden_rightleft = self.init_hidden_(batch_size, x.device)
        hidden_downup = self.init_hidden_(batch_size, x.device)

        input_lstm = x.view(x.size(0), self.opt.shape_lstm, -1)
        x_rightleft = input_lstm.permute(1, 0, 2)
        x_downup = input_lstm.permute(2, 0, 1)
        outputlr, (hidden_state_lr, cell_state_lr) = self.lstm_pose_lr(x_rightleft, hidden_rightleft)
        outputud, (hidden_state_ud, cell_state_ud) = self.lstm_pose_ud(x_downup, hidden_downup)
        final_output_lstm = torch.cat(
            (
            hidden_state_lr[0, :, :], hidden_state_lr[1, :, :], hidden_state_ud[0, :, :], hidden_state_ud[1, :, :]),
            1)
        x = self.dropout_lstm(final_output_lstm)
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)

class LSTM_first(nn.Module):
    def __init__(self,opt):
        super(LSTM_first, self).__init__()
        self.n_hidden = opt.n_hidden
        self.opt = opt
        self.lstm_pose_lr = nn.LSTM(input_size=28, hidden_size=self.n_hidden, bidirectional=True)
        self.lstm_pose_ud = nn.LSTM(input_size=28, hidden_size=self.n_hidden, bidirectional=True)
        self.fc3 = nn.Linear(opt.n_hidden*4, 10)

    def init_hidden_(self, batch_size, device):
        '''Return initialized hidden states and cell states for each biodirectional lstm cell'''
        return (torch.randn(2, batch_size, self.n_hidden).to(device),
                torch.randn(2, batch_size, self.n_hidden).to(device))


    def forward(self, x):
        batch_size = x.size()[0]
        x = np.squeeze(x)
        hidden_rightleft = self.init_hidden_(batch_size, x.device)
        hidden_downup = self.init_hidden_(batch_size, x.device)

        x_rightleft = x.permute(1, 0, 2)
        x_downup = x.permute(2, 0, 1)
        outputlr, (hidden_state_lr, cell_state_lr) = self.lstm_pose_lr(x_rightleft, hidden_rightleft)
        outputud, (hidden_state_ud, cell_state_ud) = self.lstm_pose_ud(x_downup, hidden_downup)
        output_lstm = torch.cat(
        (hidden_state_lr[0, :, :], hidden_state_lr[1, :, :], hidden_state_ud[0, :, :], hidden_state_ud[1, :, :]), 1)
        x = self.fc3(output_lstm)
        return F.log_softmax(x, dim=1)

class LSTM_only(nn.Module):
    def __init__(self,opt):
        super(LSTM_only, self).__init__()
        self.n_hidden = opt.n_hidden
        self.opt = opt
        self.lstm_pose_lr = nn.LSTM(input_size=28, hidden_size=self.n_hidden, bidirectional=True)
        self.lstm_pose_ud = nn.LSTM(input_size=28, hidden_size=self.n_hidden, bidirectional=True)
        self.lstm_fc = nn.LSTM(input_size=1, hidden_size=10, batch_first=True)

    def init_hidden_(self, batch_size, device):
        '''Return initialized hidden states and cell states for each biodirectional lstm cell'''
        return (torch.randn(2, batch_size, self.n_hidden).to(device),
                torch.randn(2, batch_size, self.n_hidden).to(device))
    def init_hidden_f_(self, batch_size, device):
        '''Return initialized hidden states and cell states for each biodirectional lstm cell'''
        return (torch.randn(1, batch_size, 10).to(device),
                torch.randn(1, batch_size, 10).to(device))

    def forward(self, x):
        batch_size = x.size()[0]
        x = np.squeeze(x)
        hidden_rightleft = self.init_hidden_(batch_size, x.device)
        hidden_downup = self.init_hidden_(batch_size, x.device)
        hidden_f = self.init_hidden_f_(batch_size, x.device)
        x_rightleft = x.permute(1, 0, 2)
        x_downup = x.permute(2, 0, 1)
        outputlr, (hidden_state_lr, cell_state_lr) = self.lstm_pose_lr(x_rightleft, hidden_rightleft)
        outputud, (hidden_state_ud, cell_state_ud) = self.lstm_pose_ud(x_downup, hidden_downup)
        output_lstm = torch.cat(
        (hidden_state_lr[0, :, :], hidden_state_lr[1, :, :], hidden_state_ud[0, :, :], hidden_state_ud[1, :, :]), 1)
        output_lstm = torch.unsqueeze(output_lstm, 2)
        outputf, (hidden_state_f, cell_statef) = self.lstm_fc(output_lstm, hidden_f)
        x = torch.squeeze(hidden_state_f)
        return F.log_softmax(x, dim=1)

class LSTM_only2(nn.Module):
    def __init__(self,opt):
        super(LSTM_only2, self).__init__()
        self.n_hidden = opt.n_hidden
        self.opt = opt
        self.lstm_pose_lr = nn.LSTM(input_size=28, hidden_size=196, bidirectional=True)
        self.lstm_pose_ud = nn.LSTM(input_size=28, hidden_size=196, bidirectional=True)
        self.lstm_fc = nn.LSTM(input_size=28, hidden_size=5, bidirectional=True, batch_first=True)

    def init_hidden_(self, batch_size, device):
        '''Return initialized hidden states and cell states for each biodirectional lstm cell'''
        return (torch.randn(2, batch_size, 196).to(device),
                torch.randn(2, batch_size, 196).to(device))
    def init_hidden_f_(self, batch_size, device):
        '''Return initialized hidden states and cell states for each biodirectional lstm cell'''
        return (torch.randn(2, batch_size, 5).to(device),
                torch.randn(2, batch_size, 5).to(device))

    def forward(self, x):
        batch_size = x.size()[0]
        x = np.squeeze(x)
        hidden_rightleft = self.init_hidden_(batch_size, x.device)
        hidden_downup = self.init_hidden_(batch_size, x.device)
        hidden_f = self.init_hidden_f_(batch_size, x.device)
        x_rightleft = x.permute(1, 0, 2)
        x_downup = x.permute(2, 0, 1)
        outputlr, (hidden_state_lr, cell_state_lr) = self.lstm_pose_lr(x_rightleft, hidden_rightleft)
        outputud, (hidden_state_ud, cell_state_ud) = self.lstm_pose_ud(x_downup, hidden_downup)
        output_lstm = torch.cat(
        (hidden_state_lr[0, :, :], hidden_state_lr[1, :, :], hidden_state_ud[0, :, :], hidden_state_ud[1, :, :]), 1)
        output_lstm= x.view(output_lstm.size(0), 28, -1)
        outputf, (hidden_state_f, cell_statef) = self.lstm_fc(output_lstm, hidden_f)
        x = torch.cat((hidden_state_f[0, :, :], hidden_state_f[1, :, :]), 1)
        return F.log_softmax(x, dim=1)



class Net_LSTM_cell_first(nn.Module):
    def __init__(self,opt):
        super(Net_LSTM_cell_first, self).__init__()
        self.n_hidden = opt.n_hidden
        self.dropout_lstm = nn.Dropout(p=opt.drop2)
        self.fc3 = nn.Linear(opt.n_hidden*4, 10)
        self.opt = opt
        self.lstm_pose_l = LSTMCell_custom(input_size=28, hidden_size=self.n_hidden, print=True)
        self.lstm_pose_r = LSTMCell_custom(input_size=28, hidden_size=self.n_hidden)
        self.lstm_pose_u = LSTMCell_custom(input_size=28, hidden_size=self.n_hidden)
        self.lstm_pose_d = LSTMCell_custom(input_size=28, hidden_size=self.n_hidden)

    def init_hidden_(self, batch_size, device):
        '''Return initialized hidden states and cell states for each biodirectional lstm cell'''
        return (torch.randn(batch_size, self.n_hidden).to(device),torch.randn(batch_size, self.n_hidden).to(device))

    def flip(self, x, dim):
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                    dtype=torch.long, device=x.device)
        return x[tuple(indices)]

    def forward(self, x):
        batch_size = x.size()[0]
        x = np.squeeze(x)

        hidden_left = self.init_hidden_(batch_size, x.device)
        hidden_right = self.init_hidden_(batch_size, x.device)
        hidden_up = self.init_hidden_(batch_size, x.device)
        hidden_down = self.init_hidden_(batch_size, x.device)

        x_left = x.permute(1, 0, 2)
        x_right = self.flip(x_left,0)
        x_up = x.permute(2, 0, 1)
        x_down = self.flip(x_up,0)
        for i in range(self.opt.shape_lstm):
            hidden_left = self.lstm_pose_l(x_left[i,:,:], hidden_left)
            hidden_right = self.lstm_pose_r(x_right[i,:,:], hidden_right)
            hidden_up = self.lstm_pose_d(x_up[i,:,:], hidden_up)
            hidden_down = self.lstm_pose_d(x_down[i,:,:], hidden_down)

        final_output_lstm = torch.cat((hidden_left[0], hidden_right[0], hidden_up[0], hidden_down[0]), 1)
        x = self.dropout_lstm(final_output_lstm)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
