#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .network_utils import *


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = layer_init(nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3, 
            stride=1,
            padding=1))

        self.conv2 = layer_init(nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3, 
            stride=1,
            padding=1))
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        return x + input


class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImpalaBlock, self).__init__()
        self.conv1 = layer_init(nn.Conv2d(
            in_channels=in_channels,
             out_channels=out_channels,
             kernel_size=3,
             stride=1))
        self.resblock1 = ResBlock(out_channels, out_channels)
        self.resblock2 = ResBlock(out_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2)(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        return x

class ImpalaBody(nn.Module):
    def __init__(self, in_channels=3, noisy_linear=False, hidden_dim=800):
        super(ImpalaBody, self).__init__()
        self.feature_dim = 256
        self.block1 = ImpalaBlock(in_channels=in_channels, out_channels=16)
        self.block2 = ImpalaBlock(in_channels=16, out_channels=32)
        self.block3 = ImpalaBlock(in_channels=32, out_channels=32)
        if noisy_linear:
            self.fc = NoisyLinear(hidden_dim, self.feature_dim)
        else:
            self.fc = layer_init(nn.Linear(hidden_dim, self.feature_dim))
        self.noisy_linear = noisy_linear
        
        #self.critic = init_critic_(nn.Linear(256, 1))
        #self.actor = init_actor_(nn.Linear(256, n_actions))
        
    def reset_noise(self):
        if self.noisy_linear:
            self.fc.reset_noise()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.relu(x)
        return x
  
class NatureConvBody(nn.Module):
    def __init__(self, in_channels=4, noisy_linear=False, hidden_dim=4096):
        super(NatureConvBody, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 128, kernel_size=8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(128, 256, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(256, 256, kernel_size=3, stride=1))
        if noisy_linear:
            self.fc4 = NoisyLinear(hidden_dim, self.feature_dim)
        else:
            self.fc4 = layer_init(nn.Linear(hidden_dim, self.feature_dim))
        self.noisy_linear = noisy_linear

    def reset_noise(self):
        if self.noisy_linear:
            self.fc4.reset_noise()

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y


class DDPGConvBody(nn.Module):
    def __init__(self, in_channels=4):
        super(DDPGConvBody, self).__init__()
        self.feature_dim = 39 * 39 * 32
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2))
        self.conv2 = layer_init(nn.Conv2d(32, 32, kernel_size=3))

    def forward(self, x):
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = y.view(y.size(0), -1)
        return y


class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu, noisy_linear=False):
        super(FCBody, self).__init__()
        dims = (state_dim,) + hidden_units
        if noisy_linear:
            self.layers = nn.ModuleList(
                [NoisyLinear(dim_in, dim_out) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        else:
            self.layers = nn.ModuleList(
                [layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])

        self.gate = gate
        self.feature_dim = dims[-1]
        self.noisy_linear = noisy_linear

    def reset_noise(self):
        if self.noisy_linear:
            for layer in self.layers:
                layer.reset_noise()

    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x


class DummyBody(nn.Module):
    def __init__(self, state_dim):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        return x
