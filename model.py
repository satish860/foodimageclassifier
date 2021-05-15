import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, 1)


class FoodImageClassifer(nn.Module):
    def __init__(self):
        super().__init__()
        mobilenet = models.mobilenet_v2(pretrained=True)
        self.body = mobilenet.features
        self.head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, 101))

    def forward(self, x):
        x = self.body(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.head(x)

    def freeze(self):
        for name, param in self.body.named_parameters():
            param.requires_grad = False