from __future__ import print_function
import torch.nn as nn
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# full-connected net to classify mnist images
class MnistDenseNet(nn.Module):
    def __init__(self):
        super(MnistDenseNet, self).__init__()
        self.mul_idx = 1

        self.layer1 = nn.Sequential(
            nn.Linear(28*28, 28*28*self.mul_idx),
            nn.Dropout(0.2),
            nn.BatchNorm1d(28*28*self.mul_idx)
        )


        self.layer2 = nn.Sequential(
            nn.Linear(28*28*self.mul_idx, 28*28*self.mul_idx),
            nn.Dropout(0.2),
            nn.BatchNorm1d(28*28*self.mul_idx)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(28*28*self.mul_idx, 10),
            nn.Dropout(0.2),
            nn.BatchNorm1d(10)
        )

        self.logsoftmax = nn.LogSoftmax()



    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return self.logsoftmax(x)

def train(args, model, device, train_loader, optimizer, epoch, train_accuracy=None):
    model.train()
    correct = 0
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
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    if train_accuracy is not None:
        train_accuracy.append(100. * correct / len(train_loader.dataset))

    print('\nTrain set: Accuracy: {}/{} ({:.0f}%)'.format(
        correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))



def test(args, model, device, test_loader, test_accuracy=None):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    if test_accuracy is not None:
        test_accuracy.append(100. * correct / len(test_loader.dataset))

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def get_target(num):
    '''
    Transforms target number into vector length of 10
    For example:
    For number 3 vector is [-1, -1, -1, 1, -1, -1, -1, -1, -1, -1,]
    :return: torch.Tensor
    '''

    t = torch.zeros(10)-1.
    t[num] = 1.

    return t

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

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

    model = MnistDenseNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    test_accuracy = []
    train_accuracy = []

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader, test_accuracy)

    if (args.save_model):
        torch.save(model.state_dict(), "mnist_fc_net.pt")

    print("test_accuracy\n", test_accuracy)
    print("train_accuracy\n", train_accuracy)


if __name__ == '__main__':
    main()
