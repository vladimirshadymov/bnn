from __future__ import print_function
import csv
from itertools import zip_longest
import torch.nn as nn
from bnn_modules import BinarizedLinear, BinarizeFunction
from training_routines import train, test
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms

class MnistDenseBNN(nn.Module):
    def __init__(self):
        super(MnistDenseBNN, self).__init__()
        self.mul_idx = 2

        self.layer1 = nn.Sequential(
            BinarizedLinear(28*28, 512*self.mul_idx),
            nn.BatchNorm1d(512*self.mul_idx)
        )
        self.sign1 = BinarizeFunction.apply
        self.dp1 = nn.Dropout(0.5)


        self.layer2 = nn.Sequential(
            BinarizedLinear(512*self.mul_idx, 512*self.mul_idx),
            nn.BatchNorm1d(512*self.mul_idx)
        )
        self.sign2 = BinarizeFunction.apply
        self.dp2 = nn.Dropout(0.5)

        self.layer3 = nn.Sequential(
            BinarizedLinear(512*self.mul_idx, 10),
            nn.BatchNorm1d(10)
        )
        self.sign3 = BinarizeFunction.apply
        self.dp3 = nn.Dropout(0.2)


    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.dp1(self.sign1(self.layer1(x)))
        x = self.dp2(self.sign2(self.layer2(x)))
        x = self.dp3(self.layer3(x))
        return x

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

    model = MnistDenseBNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-8)

    test_accuracy = []
    train_accuracy = []

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader, train_loader, test_accuracy, train_accuracy)

    if (args.save_model):
        torch.save(model.state_dict(), "../model/mnist_bnn.pt")

    d = [train_accuracy, test_accuracy]
    export_data = zip_longest(*d, fillvalue='')
    with open('../model/mnist_bnn_report.csv', 'w', encoding="ISO-8859-1", newline='') as report_file:
        wr = csv.writer(report_file)
        wr.writerow(("Train accuracy", "Test accuracy"))
        wr.writerows(export_data)
    report_file.close()

if __name__ == '__main__':
    main()
