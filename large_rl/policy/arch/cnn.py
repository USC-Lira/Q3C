import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, img_size=7, dim_in_channels=3, channels='32_64_32'):
        super(CNN, self).__init__()
        kernel_size1 = 2
        stride1 = 1
        # channels = [32, 64, 32]
        channels = channels.split('_')
        channels = list(map(int, channels))
        self.conv1 = nn.Conv2d(dim_in_channels, channels[0], kernel_size=kernel_size1, stride=stride1)
        # self.bn1 = nn.BatchNorm2d(32)
        img_size = (img_size - kernel_size1 + stride1) // stride1
        kernel_size2 = 2
        stride2 = 1
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=kernel_size2, stride=stride2)
        img_size = (img_size - kernel_size2 + stride2) // stride2
        if img_size == 1:
            kernel_size3 = 1
        elif img_size % 2:
            kernel_size3 = 3
        else:
            kernel_size3 = 2
        stride3 = 2
        # self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size=kernel_size3, stride=stride3)
        img_size = (img_size - kernel_size3 + stride3) // stride3
        # self.bn3 = nn.BatchNorm2d(32)
        # self.fc3 = nn.Linear(32, 64)
        self.dim_out = channels[2] * img_size * img_size

        # This is found from my repo that i used to work on. so reference, sorry.
        nn.init.orthogonal_(self.conv1.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv2.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv3.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.conv1.bias.data, 0)
        nn.init.constant_(self.conv2.bias.data, 0)
        nn.init.constant_(self.conv3.bias.data, 0)


    def forward(self, x):
        batch, candidate, channel, H, W = x.size()
        x = x.view(batch * candidate, channel, H, W)
        x = x.permute((0, 3, 1, 2))  # (batch * candidate) x channel x H x W
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(-1, self.dim_out)
        return x.view(batch, candidate, x.shape[-1])


class OldCNN(nn.Module):
    def __init__(self, img_size=7, dim_in_channels=3, dim_out=16):
        super(OldCNN, self).__init__()
        if img_size % 2:
            kernel_size = 3
        else:
            kernel_size = 4
        stride = 2
        self.conv1 = nn.Conv2d(dim_in_channels, 16, kernel_size=kernel_size, stride=stride)
        img_size = (img_size - kernel_size + stride) // stride
        self.conv2 = nn.Conv2d(16, 32, kernel_size=img_size, stride=1)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, dim_out)
        self.dim_out = dim_out

    def forward(self, x):
        batch, candidate, channel, H, W = x.size()
        x = x.view(batch * candidate, channel, H, W)
        x = x.permute((0, 3, 1, 2))  # (batch * candidate) x channel x H x W
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x.view(batch, candidate, x.shape[-1])


class FlatMLP(nn.Module):
    def __init__(self, img_size=7, dim_in_channels=3, dim_out=16):
        super(FlatMLP, self).__init__()
        self.fc1 = nn.Linear(img_size * img_size * dim_in_channels, 64)
        self.fc2 = nn.Linear(64, dim_out)
        self.dim_out = dim_out

    def forward(self, x):
        batch, candidate, channel, H, W = x.size()
        x = x.view(batch * candidate, channel * H * W)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(batch, candidate, x.shape[-1])

class Flat(nn.Module):
    def __init__(self, img_size=7, dim_in_channels=3):
        super(Flat, self).__init__()
        self.fc1 = nn.Linear(5, 5)
        self.dim_out = img_size * img_size * dim_in_channels

    def forward(self, x):
        batch, candidate, channel, H, W = x.size()
        return x.view(batch, candidate, channel * H * W)


def _test():
    import torch
    num_envs, num_candidates, h, w, c = 4, 10, 7, 7, 9
    obs = torch.rand(num_envs, num_candidates, h, w, c)

    cnn = CNN(dim_in_channels=c)
    out = cnn(obs)
    print(obs.shape, out.shape)

    cnn = OldCNN(dim_in_channels=c)
    out = cnn(obs)
    print(obs.shape, out.shape)

    cnn = FlatMLP(dim_in_channels=c)
    out = cnn(obs)
    print(obs.shape, out.shape)


if __name__ == '__main__':
    _test()
