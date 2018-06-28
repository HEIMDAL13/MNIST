import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self,opt):

        super(Net, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(784, opt.first_size),nn.ReLU(),nn.Dropout(p=opt.drop1))
        self.fc2 = nn.Sequential(nn.Linear(opt.first_size, opt.n_hidden*4),nn.ReLU(),nn.Dropout(p=opt.drop2))
        self.fc3 = nn.Linear(opt.n_hidden*4, 10)
        self.opt = opt
    def forward(self, x):
        input_inter = x.view(x.size(0), -1)
        x = self.fc1(input_inter)
        x = self.fc2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class Net_LSTM(nn.Module):
    def __init__(self,opt):
        super(Net_LSTM, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(784, opt.first_size),nn.ReLU(),nn.Dropout(p=opt.drop1))
        self.dropout_lstm = nn.Dropout(p=opt.drop2)
        self.fc3 = nn.Linear(opt.n_hidden*4, 10)
        self.opt = opt
        if opt.shape_lstm == 0:
            self.lstm_pose_lr = nn.LSTM(input_size=1, hidden_size=opt.n_hidden*2, bidirectional=True)
        else:
            self.lstm_pose_lr = nn.LSTM(input_size=int(opt.first_size/opt.shape_lstm), hidden_size=opt.n_hidden, bidirectional=True)
            self.lstm_pose_ud = nn.LSTM(input_size=opt.shape_lstm, hidden_size=opt.n_hidden, bidirectional=True)
        print("1: FC 784 x",opt.first_size)

    def forward(self, x):
        input_inter = x.view(x.size(0), -1)
        x = self.fc1(input_inter)
        if self.opt.shape_lstm == 0:
            input_lstm = x.view(x.size(0), 1, -1)
            outputlr, (hidden_state_lr, cell_state_lr) = self.lstm_pose_lr(input_lstm.permute(2, 0, 1))
            final_output_lstm = torch.cat((hidden_state_lr[0, :, :], hidden_state_lr[1, :, :]), 1)
        else:
            input_lstm = x.view(x.size(0), self.opt.shape_lstm, -1)
            outputlr, (hidden_state_lr, cell_state_lr) = self.lstm_pose_lr(input_lstm.permute(1, 0, 2))
            outputud, (hidden_state_ud, cell_state_ud) = self.lstm_pose_ud(input_lstm.permute(2, 0, 1))
            final_output_lstm = torch.cat(
            (hidden_state_lr[0, :, :], hidden_state_lr[1, :, :], hidden_state_ud[0, :, :], hidden_state_ud[1, :, :]), 1)
        x = self.dropout_lstm(final_output_lstm)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)