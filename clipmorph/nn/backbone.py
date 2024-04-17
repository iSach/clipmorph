from torch import nn
import torchvision as tv
from torchvision.models import VGG19_Weights
from wandb.wandb_torch import torch


class Vgg19(nn.Module):
    def __init__(self, device='cpu'):
        super(Vgg19, self).__init__()
        vgg_features = tv.models.vgg19(weights=VGG19_Weights.DEFAULT).features
        self.block1 = nn.Sequential()
        self.block2 = nn.Sequential()
        self.block3 = nn.Sequential()
        self.block4 = nn.Sequential()
        for i in range(4):
            self.block1.add_module(str(i), vgg_features[i])
        for i in range(4, 9):
            self.block2.add_module(str(i), vgg_features[i])
        for i in range(9, 16):
            self.block3.add_module(str(i), vgg_features[i])
        for i in range(16, 23):
            self.block4.add_module(str(i), vgg_features[i])

        for param in self.parameters():
            param.requires_grad = False

        self.mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).to(device)

        self.mean = self.mean.view(-1, 1, 1)
        self.std = self.std.view(-1, 1, 1)

    def forward(self, x):
        # Return outputs of ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
        vgg_outputs = []
        y = self.block1(x)
        vgg_outputs.append(y)
        y = self.block2(y)
        vgg_outputs.append(y)
        y = self.block3(y)
        vgg_outputs.append(y)
        y = self.block4(y)
        vgg_outputs.append(y)

        return vgg_outputs

    def normalize_batch(self, batch):
        batch = batch / 255.0
        return (batch - self.mean) / self.std
