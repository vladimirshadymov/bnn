from __future__ import print_function
import torch.nn as nn
import csv
from itertools import zip_longest
import argparse
import torch
from training_routines import train, test
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

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
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
        datasets.MNIST('../../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = MnistDenseNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    test_accuracy = []
    train_accuracy = []

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader, train_loader, test_accuracy, train_accuracy)

    d = [train_accuracy, test_accuracy]
    export_data = zip_longest(*d, fillvalue='')
    with open('../../model/mnist_fc_net_report.csv', 'w', encoding="ISO-8859-1", newline='') as report_file:
        wr = csv.writer(report_file)
        wr.writerow(("Train accuracy", "Test accuracy"))
        wr.writerows(export_data)
    report_file.close()

    if (args.save_model):
        torch.save(model.state_dict(), "../../model/mnist_fc_net.pt")

if __name__ == '__main__':
    main()
