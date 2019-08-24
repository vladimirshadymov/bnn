from __future__ import print_function
import csv
from itertools import zip_longest
import torch.nn as nn
from bnn_modules import BinarizedLinear, SignEst, BinarizedConv2d
from training_routines import train, test
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms

class Cifar10ConvBNN(nn.Module):
    def __init__(self):
        super(Cifar10ConvBNN, self).__init__()
        self.sign = SignEst.apply
        self.pool = nn.MaxPool2d(2)

        self.conv1 = BinarizedConv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = BinarizedConv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = BinarizedConv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = BinarizedConv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = BinarizedConv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        self.conv6 = BinarizedConv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(512)

        self.binlin1 = BinarizedLinear(512*4*4, 1024)
        self.BN1 = nn.BatchNorm1d(1024)

        self.binlin2 = BinarizedLinear(1024, 1024)
        self.BN2 = nn.BatchNorm1d(1024)

        self.binlin3 = BinarizedLinear(1024, 10)
        self.BN3 = nn.BatchNorm1d(10)

    def forward(self, x):
        x = self.sign(self.bn1(self.conv1(x)))
        x = self.sign(self.pool(self.bn2(self.conv2(x))))
        x = self.sign(self.bn3(self.conv3(x)))
        x = self.sign(self.pool(self.bn4(self.conv4(x))))
        x = self.sign(self.bn5(self.conv5(x)))
        x = self.sign(self.pool(self.bn6(self.conv6(x))))
        x = x.view(-1, 4*4*512)
        x = self.sign(self.BN1(self.binlin1(x)))
        x = self.sign(self.BN2(self.binlin2(x)))
        x = self.BN3(self.binlin3(x))
        return x

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 BNN')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
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
    parser.add_argument('--weight-decay', type=float, default=0, metavar='W',
                        help='coefficient of L2 regulariztion')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../../data/', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../../data/', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Cifar10ConvBNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    test_accuracy = []
    train_accuracy = []

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader, train_loader, test_accuracy, train_accuracy)

    if (args.save_model):
        torch.save(model.state_dict(), "../../model/cifar10_conv_bnn.pt")

    d = [train_accuracy, test_accuracy]
    export_data = zip_longest(*d, fillvalue='')
    with open('../../model/cifar10_conv_bnn_report.csv', 'w', encoding="ISO-8859-1", newline='') as report_file:
        wr = csv.writer(report_file)
        wr.writerow(("Train accuracy", "Test accuracy"))
        wr.writerows(export_data)
    report_file.close()

if __name__ == '__main__':
    main()