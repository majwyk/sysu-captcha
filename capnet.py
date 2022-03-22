import torch
import numpy as np
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class CapNet(nn.Module):
    def __init__(self):
        super(CapNet, self).__init__()
        # 多层fc
        self.conv1 = nn.Conv2d(1, 3, 3)
        self.conv2 = nn.Conv2d(3, 6, 3)
        self.fc1 = nn.Linear(2880, 1024)
        self.fc2 = nn.Linear(1024, 64)
        self.fc3 = nn.Linear(64, 25)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def train(epoch_num, train_loader):
    size = len(train_loader.dataset)
    model.train()
    for epoch in range(epoch_num):
        epoch = epoch + 1
        correct = 0
        print("Training... Epoch = %d" % epoch)
        for batch, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            loss = lossfn(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 10 == 0:
                loss, current = loss.item(), batch * len(data)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        print('epoch: %d, loss: %f, correct: %f' % (epoch, loss.item(), correct.item()))
    torch.save(model, './utils/captcha/model.pkl')


def test(path, test_loader):
    size = len(test_loader.dataset)
    model = torch.load(path)
    model.eval()
    correct = 0
    for _, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)

        output = model(data)
        pred = output.data.max(dim=1)[1]
        correct += pred.eq(target.data).cpu()

    print('test correct: %f' % (size / correct.item()))


def apply(path, device, data):
    model = torch.load(path)
    model.to(device)
    model.eval()
    resize = transforms.Resize((34, 20))
    normalize = transforms.Normalize(0.5, 0.5)
    ch1 = resize(torch.from_numpy(data[:,3:23]).unsqueeze(0))
    ch2 = resize(torch.from_numpy(data[:,23:44]).unsqueeze(0))
    ch3 = resize(torch.from_numpy(data[:,44:64]).unsqueeze(0))
    ch4 = resize(torch.from_numpy(data[:,64:85]).unsqueeze(0))
    data = torch.cat((ch1, ch2, ch3, ch4), dim=0).unsqueeze(1)
    data = normalize(data.float()).to(device)

    output = model(data)
    pred = output.data.max(dim=1)[1]

    return pred.cpu().numpy().tolist()


if __name__ == '__main__':
    epoch = 3
    model_path = './utils/captcha/model.pkl'

    use_cuda = torch.cuda.is_available() and True
    device = torch.device("cuda" if use_cuda else "cpu")

    # Dataset
    trainset = datasets.ImageFolder(r'./utils/captcha/train', transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((34, 20)),
        transforms.Grayscale(1),
        transforms.Normalize(0.5, 0.5)
    ]))
    train_loader = DataLoader(trainset, batch_size=64,
                              shuffle=True, num_workers=4)

    testset = datasets.ImageFolder(r'./utils/captcha/test', transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((34, 20)),
        transforms.Grayscale(1),
        transforms.Normalize(0.5, 0.5)
    ]))
    test_loader = DataLoader(testset, batch_size=1,
                             shuffle=True, num_workers=4)

    # Model
    model = CapNet().to(device)
    lossfn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    train(epoch_num=epoch, train_loader=train_loader)
    test(path=model_path, test_loader=test_loader)
