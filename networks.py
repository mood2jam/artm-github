import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# The VGG network was taken from https://github.com/kuangliu/pytorch-cifar

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name='VGG11'):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(nn.Linear(512, 256),
                                        nn.PReLU(),
                                        nn.Linear(256, 10))
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

# The following networks were modified from https://github.com/adambielski/siamese-triplet

class EmbeddingNet(nn.Module):
    def __init__(self, in_channels=1, adjusting_constant=4):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(in_channels, 32, 5), nn.PReLU(), #32@28*28
                                     nn.MaxPool2d(2, stride=2), #32@14*14
                                     nn.Conv2d(32, 64, 5), nn.PReLU(), #64@10*10
                                     nn.MaxPool2d(2, stride=2)) #64@5*5

        self.fc = nn.Sequential(nn.Linear(64 * adjusting_constant * adjusting_constant, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 10),
                                nn.BatchNorm1d(10)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)

class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)

class Generator(nn.Module):
    def __init__(self, out_channels=1, adjusting_constant=4):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(nn.Linear(10, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 64 * adjusting_constant * adjusting_constant)
                                )
        self.convnet = nn.Sequential(nn.ConvTranspose2d(64, 32, 5), nn.PReLU(),  # 64@10*10
                                     nn.ConvTranspose2d(32, 32, 5), nn.PReLU(),  # 64@10*10
                                     nn.ConvTranspose2d(32, 16, 5), nn.PReLU(),  # 64@10*10
                                     nn.ConvTranspose2d(16, 16, 5), nn.PReLU(),  # 64@10*10
                                     nn.ConvTranspose2d(16, 8, 5), nn.PReLU(),  # 64@10*10
                                     nn.ConvTranspose2d(8, 8, 3), nn.PReLU(),  # 64@10*10
                                     nn.ConvTranspose2d(8, out_channels, 3), nn.PReLU())  # 32@28*28

    def forward(self, x):
        output = self.fc(x)
        output = output.view(-1, 64, 4, 4)
        output = self.convnet(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)

# Used for the bottom two networks only
args = {
    'epochs': 500,
    'width': 32,
    'latent_width': 4,
    'depth': 16,
    'advdepth': 16,
    'advweight': 0.5,
    'reg': 0.2,
    'latent': 10,
    'colors': 1,
    'lr': 0.0001,
    'batch_size': 64,
    'device': 'cuda:0'
}
scales = int(round(math.log(args['width'] // args['latent_width'], 2)))

class Encoder(nn.Module):
    def __init__(self, scales=scales, depth=args['depth'], latent=args['latent'], colors=args['colors']):
        super(Encoder, self).__init__()
        layers = []
        layers.append(nn.Conv2d(colors, depth, 1, padding=1))
        kp = depth
        for scale in range(scales):
            k = depth << scale
            layers.extend([nn.Conv2d(kp, k, 3, padding=1), nn.ReLU()])
            layers.extend([nn.Conv2d(k, k, 3, padding=1), nn.ReLU()])
            layers.append(nn.MaxPool2d(2))
            kp = k
        k = depth << scales
        layers.extend([nn.Conv2d(kp, k, 3, padding=1), nn.ReLU()])
        layers.append(nn.Conv2d(k, latent, 3, padding=0))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, args['latent'])
        return x

class Decoder(nn.Module):
    def __init__(self, scales=scales+2, depth=args['depth'], latent=args['latent'], colors=args['colors']):
        super(Decoder, self).__init__()
        layers = []
        kp = latent
        for scale in range(scales - 1, -1, -1):
            k = depth << scale
            layers.extend([nn.Conv2d(kp, k, 3, padding=1), nn.ReLU()])
            layers.extend([nn.Conv2d(k, k, 3, padding=1), nn.ReLU()])
            layers.append(nn.Upsample(scale_factor=2))
            kp = k
        layers.extend([nn.Conv2d(kp, depth, 3, padding=0), nn.ReLU()])
        layers.append(nn.Conv2d(depth, colors, 3, padding=0))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, args['latent'], 1, 1)
        x = self.net(x)
        return x

def get_embedding_net(params):
    if params['dset'] == 'MNIST' or params['dset'] == 'FASHIONMNIST':
        # model = EmbeddingNet(in_channels=1, adjusting_constant=4)
        model = Encoder()
    elif params['dset'] == 'CIFAR10':
        model = EmbeddingNet(in_channels=3, adjusting_constant=5)

    return model



