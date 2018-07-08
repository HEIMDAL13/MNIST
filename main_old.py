from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.sampler import Sampler

class SubsetSampler(Sampler):  # pylint: disable=too-few-public-methods
    """
    Return subset of dataset. For example to enforce overfitting.
    """

    def __init__(self, indices, subset_size, random_subset=False, shuffle=True):
        assert subset_size <= len(indices), (
            f"The subset size ({subset_size}) must be smaller "
            f"or equal to the sampler size ({len(indices)}).")
        self._subset_size = subset_size
        self._shuffle = shuffle
        self._random_subset = random_subset
        self._indices = indices
        self._subset = None
        self.set_subset()

    def set_subset(self):
        """Set subset from sampler with size self._subset_size"""
        if self._random_subset:
            perm = torch.randperm(len(self._indices))
            self._subset = self._indices[perm][:self._subset_size]
        else:
            self._subset = torch.Tensor(self._indices[:self._subset_size])

    def __iter__(self):
        """Iterate over same or shuffled subset."""
        if self._shuffle:
            perm = torch.randperm(self._subset_size)
            return iter(self._subset[perm].tolist())
        return iter(self._subset)

    def __len__(self):
        return len(self._subset)



class Net(nn.Module):
    def __init__(self,args):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, args.first_size)
        self.fc2 = nn.Linear(args.first_size, args.n_hidden*4)
        self.fc3 = nn.Linear(args.n_hidden*4, 10)
        self.args = args

        print("1: FC 784 x", args.first_size)
        print("2: FC ", args.first_size, "x", args.n_hidden*4)
        print("3: FC ", args.n_hidden*4, "x", 10)
    def forward(self, x):
        input_inter = x.view(x.size(0), -1)
        x = F.relu(self.fc1(input_inter))
        x = F.relu(self.fc2(x))
        if self.args.dropout == 1:
            x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class Net_LSTM(nn.Module):
    def __init__(self,args):
        super(Net_LSTM, self).__init__()

        self.fc1 = nn.Linear(784, args.first_size)
        self.fc3 = nn.Linear(args.n_hidden*4, 10)
        self.args = args
        if args.shape_lstm == 0:
            self.lstm_pose_lr = nn.LSTM(input_size=1, hidden_size=args.n_hidden*2, bidirectional=True)
        else:
            self.lstm_pose_lr = nn.LSTM(input_size=int(args.first_size/args.shape_lstm), hidden_size=args.n_hidden, bidirectional=True)
            self.lstm_pose_ud = nn.LSTM(input_size=args.shape_lstm, hidden_size=args.n_hidden, bidirectional=True)
        print("1: FC 784 x",args.first_size)
        if args.shape_lstm==0:
            print("2: LSTM ",args.first_size, "x", args.n_hidden*4," Shaped as:", args.first_size,"x 1"," n_hidden",args.n_hidden*2)
        else:
            print("2: LSTM ",args.first_size, "x", args.n_hidden*4," Shaped as:",args.shape_lstm,"x", int(args.first_size/args.shape_lstm)," n_hidden",args.n_hidden)
        print("3: FC ",args.n_hidden*4,"x",10)

    def forward(self, x):
        input_inter = x.view(x.size(0), -1)
        x = F.relu(self.fc1(input_inter))
        if self.args.shape_lstm == 0:
            input_lstm = x.view(x.size(0), 1, -1)
            outputlr, (hidden_state_lr, cell_state_lr) = self.lstm_pose_lr(input_lstm.permute(2, 0, 1))
            final_output_lstm = torch.cat((hidden_state_lr[0, :, :], hidden_state_lr[1, :, :]), 1)
        else:
            input_lstm = x.view(x.size(0), self.args.shape_lstm, -1)
            outputlr, (hidden_state_lr, cell_state_lr) = self.lstm_pose_lr(input_lstm.permute(1, 0, 2))
            outputud, (hidden_state_ud, cell_state_ud) = self.lstm_pose_ud(input_lstm.permute(2, 0, 1))
            final_output_lstm = torch.cat(
            (hidden_state_lr[0, :, :], hidden_state_lr[1, :, :], hidden_state_ud[0, :, :], hidden_state_ud[1, :, :]), 1)
        if self.args.dropout == 1:
            x = F.dropout(final_output_lstm, training=self.training)
        else:
            x = final_output_lstm
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch,file):
    losses =[]
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            file.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item())+"\n")

            losses.append(loss.item())
    return losses
def test(args, model, device, test_loader, file):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    file.write('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))+"\n")

    return test_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--first_size', type=int, default=128,
                        help='size first fc layer')
    parser.add_argument('--n_hidden', type=int, default=0,
                        help='hidden_lstm layers')
    parser.add_argument('--shape_lstm', type=int, default=0,
                        help='shape lstm')
    parser.add_argument('--lstm', type=int, default=0,
                        help='lstm')
    parser.add_argument('--name', type=str, default=0,
                        help='name')
    parser.add_argument('--dropout', type=int, default=1,
                        help='name')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # subset_train = SubsetSampler(train_loader,500)
    # subset_train.set_subset()
    # train_loader = subset_train

    if args.lstm == 0:
        model = Net(args).to(device)
    else:
        model = Net_LSTM(args).to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters: ", pytorch_total_params)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    losses_train = []
    losses_test = []
    stds = []
    file = open("results_"+args.name+".txt", "w")
    if args.lstm ==0:
        file.write("PARAMETERS: \n")
        file.write("1: FC 784x"+str(args.first_size)+"\n")
        file.write("2: FC "+str(args.first_size)+ "x"+str(args.n_hidden * 4)+"\n")
        file.write("3: FC "+str(args.n_hidden * 4)+ "x10\n")
    else:
        file.write("1: FC 784x"+str(args.first_size)+"\n")
        if args.shape_lstm==0:
            file.write("2: LSTM "+str(args.first_size)+"x"+str(args.n_hidden*4)+" Shaped as:"+str(args.first_size)+"x 1"+" n_hidden"+str(args.n_hidden*2)+"\n")
        else:
            file.write("2: LSTM "+str(args.first_size)+ "x"+str( args.n_hidden*4)+" Shaped as:"+str(args.shape_lstm)+"x"+str(int(args.first_size/args.shape_lstm))+" n_hidden"+str(args.n_hidden)+"\n")
            file.write("3: FC "+str(args.n_hidden*4)+"x10\n")
    file.write("\n\nTotal parameters: "+str(pytorch_total_params)+"\n\n")
    argx = vars(args)
    file.write('------------ Options -------------\n')
    for k, v in sorted(argx.items()):
        file.write('%s: %s' % (str(k), str(v)))
        file.write("\n")
    file.write('-------------- End ----------------\n\n')
    for epoch in range(1, args.epochs + 1):
        losses = train(args, model, device, train_loader, optimizer, epoch,file)
        train_std = np.std(losses)
        train_loss = np.average(losses)
        test_loss = test(args, model, device, test_loader,file)
        losses_train.append(train_loss)
        losses_test.append(test_loss)
        stds.append(train_std)
    batches = np.arange(1,len(losses_train)+1)
    best_loss_test = np.amin(losses_test)
    best_loss_epoch = np.argmin(losses_test)+1
    plt.plot(batches, losses_train, batches, losses_test)
    plt.ylabel('loss')
    plt.title('train/test loss')
    plt.savefig('plot_'+args.name+'.jpg')
    #plt.show()

    file.write("\n\nBest test loss: {0:.4f}  Epoch: {1:}".format(best_loss_test,best_loss_epoch))

    file.close()

if __name__ == '__main__':
    main()