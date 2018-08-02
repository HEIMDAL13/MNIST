import torch.nn as nn
import torch.nn.functional as F
import torch

class Net_2FC(nn.Module):
    def __init__(self,opt):

        super(Net_2FC, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(opt.input_size, opt.first_size),nn.ReLU(),nn.Dropout(p=opt.drop1))
        self.fc2 = nn.Sequential(nn.Linear(opt.first_size, opt.n_hidden*4),nn.ReLU(),nn.Dropout(p=opt.drop2))
        self.batch = nn.BatchNorm1d(opt.n_hidden*4)
        self.fc3 = nn.Linear(opt.n_hidden*4, 10)
        self.opt = opt
    def forward(self, x):
        input_inter = x.view(x.size(0), -1)
        x = self.fc1(input_inter)
        x = self.fc2(x)
        if self.opt.batch_norm == 1:
            x = self.batch(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class Net_RNN(nn.Module):
    def __init__(self,opt):
        super(Net_RNN, self).__init__()

        self.n_hidden = opt.n_hidden
        self.fc1 = nn.Sequential(nn.Linear(opt.input_size, opt.first_size), nn.ReLU(), nn.Dropout(p=opt.drop1))
        self.dropout_lstm = nn.Dropout(p=opt.drop2)
        self.fc3 = nn.Linear(opt.n_hidden * 4, 10)
        self.opt = opt
        self.rnn_pose_lr = nn.RNN(input_size=36, hidden_size=self.n_hidden, bidirectional=True)
        self.rnn_pose_ud = nn.RNN(input_size=36, hidden_size=self.n_hidden, bidirectional=True)

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


class Net_LSTM(nn.Module):
    def __init__(self,opt):
        super(Net_LSTM, self).__init__()
        self.n_hidden = opt.n_hidden
        self.fc1 = nn.Sequential(nn.Linear(opt.input_size, opt.first_size),nn.ReLU(),nn.Dropout(p=opt.drop1))
        self.dropout_lstm = nn.Dropout(p=opt.drop2)
        self.fc3 = nn.Linear(opt.n_hidden*4, 10)
        self.opt = opt
        if opt.shape_lstm == 0:
            self.n_hidden = self.n_hidden*2
            self.lstm_pose_lr = nn.LSTM(input_size=1, hidden_size=self.n_hidden,bidirectional=True)
        else:
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

class Net_LSTM2(nn.Module):
    def __init__(self, opt):
        super(Net_LSTM2, self).__init__()
        self.n_hidden = opt.n_hidden
        self.fc1 = nn.Sequential(nn.Linear(opt.input_size, 1056), nn.ReLU(), nn.Dropout(p=opt.drop1))
        self.fc2 = nn.Sequential(nn.Linear(1056, 1024), nn.ReLU(), nn.Dropout(p=opt.drop2))
        self.dropout_lstm = nn.Dropout(p=opt.drop2)
        self.fc3 = nn.Linear(opt.n_hidden * 4, 10)
        self.opt = opt
        if opt.shape_lstm == 0:
            self.n_hidden = self.n_hidden * 2
            self.lstm_pose_lr = nn.LSTM(input_size=1, hidden_size=self.n_hidden, bidirectional=True)
        else:
            self.lstm_pose_lr = nn.LSTM(input_size=int(opt.first_size / opt.shape_lstm), hidden_size=self.n_hidden,
                                        bidirectional=True)
            self.lstm_pose_ud = nn.LSTM(input_size=opt.shape_lstm, hidden_size=self.n_hidden, bidirectional=True)

    def init_hidden_(self, batch_size, device):
        '''Return initialized hidden states and cell states for each biodirectional lstm cell'''
        return (torch.randn(2, batch_size, self.n_hidden).to(device),
                torch.randn(2, batch_size, self.n_hidden).to(device))


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


class LeNet(nn.Module):
    def __init__(self,opt):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(p=opt.drop1)
        self.fc_drop = nn.Dropout(p=opt.drop2)
        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 1000)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = self.fc_drop(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class LeNet_LSTM(nn.Module):
    def __init__(self,opt):
        super(LeNet_LSTM, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(p=opt.drop1)
        self.fc_drop = nn.Dropout(p=opt.drop2)
        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
        self.opt = opt
        self.n_hidden = opt.n_hidden

        if opt.shape_lstm == 0:
            self.n_hidden = self.n_hidden * 2
            self.lstm_pose_lr = nn.LSTM(input_size=1, hidden_size=self.n_hidden, bidirectional=True)
        else:
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
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)

        hidden_rightleft = self.init_hidden_(batch_size, x.device)
        hidden_downup = self.init_hidden_(batch_size, x.device)


        if self.opt.shape_lstm == 0:
            input_lstm = x.view(x.size(0), 1, -1)
            x_rightleft = input_lstm.permute(2, 0, 1)
            outputlr, (hidden_state_lr, cell_state_lr) = self.lstm_pose_lr(x_rightleft, hidden_rightleft)
            final_output_lstm = torch.cat((hidden_state_lr[0, :, :], hidden_state_lr[1, :, :]), 1)
        else:
            input_lstm = x.view(x.size(0), self.opt.shape_lstm, -1)
            x_rightleft = input_lstm.permute(1, 0, 2)
            x_downup = input_lstm.permute(2, 0, 1)
            outputlr, (hidden_state_lr, cell_state_lr) = self.lstm_pose_lr(x_rightleft, hidden_rightleft)
            outputud, (hidden_state_ud, cell_state_ud) = self.lstm_pose_ud(x_downup, hidden_downup)
            final_output_lstm = torch.cat(
                (
                hidden_state_lr[0, :, :], hidden_state_lr[1, :, :], hidden_state_ud[0, :, :], hidden_state_ud[1, :, :]),
                1)
        x = self.fc_drop(x)
        x = self.fc3(final_output_lstm)

        return F.log_softmax(x, dim=1)

class LeNet_plus(nn.Module):
    def __init__(self, opt):
        super(LeNet_plus, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc_drop = nn.Dropout(p=opt.drop2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_drop(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
