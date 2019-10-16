import torch
import torch.nn.functional as F
from bnn_modules import hinge_p_loss

def train(args, model, device, train_loader, optimizer, epoch, penalty='cross_entropy'):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if penalty == 'cross_entropy':
            loss = F.cross_entropy(output, target, reduction='mean')
        elif penalty == 'multi_margin':
            loss = F.multi_margin_loss(output, target, p=1, reduction='mean')
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader, train_loader=None, test_accuracy=None, train_accuracy=None, penalty='cross_entropy'):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if penalty == 'cross_entropy':
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
            elif penalty == 'multi_margin':
                test_loss += F.multi_margin_loss(output, target, p=1, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    if test_accuracy is not None:
        test_accuracy.append(100. * correct / len(test_loader.dataset))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    if not train_loader is None:
        correct = 0
        train_loss = 0
        with torch.no_grad():
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                if penalty == 'cross_entropy':
                    train_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                elif penalty == 'multi_margin':
                    train_loss += F.multi_margin_loss(output, target, p=1, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        train_loss /= len(train_loader.dataset)

        if train_accuracy is not None:
            train_accuracy.append(100. * correct / len(train_loader.dataset))

        print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            train_loss, correct, len(train_loader.dataset),
            100. * correct / len(train_loader.dataset)))