import torch.nn as nn
import torch.nn.functional as F
import torch

from models.buildingblocks import MLP, SharedMLP, debug
# from buildingblocks import *


class T_NET(nn.Module):
    def __init__(self, input_channel, output_shape) -> None:
        super().__init__()
        self.input_channel = input_channel
        self.output_shape = output_shape
        self.layers = nn.Sequential(
            SharedMLP(input_channel, 64),
            SharedMLP(64, 128),
            SharedMLP(128, 1024),
            nn.AdaptiveMaxPool1d(1),
            MLP(1024, 512),
            MLP(512, 256),
            nn.Linear(256, output_shape**2),
        )
        nn.init.constant_(self.layers[-1].weight, 0)
        with torch.no_grad():
            self.layers[-1].bias = nn.Parameter(torch.eye(output_shape).flatten(), True)

    def forward(self, x: torch.tensor):
        x = self.layers(x.transpose(1, 2)).reshape(-1, self.output_shape, self.output_shape)
        return x


class PointNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first_transform = T_NET(3, 3)
        self.second_transform = T_NET(64, 64)
        self.first_group_of_MLPs = nn.Sequential(
            SharedMLP(3, 64)
        )
        self.second_group_of_MLPs = nn.Sequential(
            SharedMLP(64, 128),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024)
        )
        self.maxpool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        t1 = self.first_transform(x)
        x = torch.matmul(x, t1).transpose(1, 2)
        x = self.first_group_of_MLPs(x).transpose(1, 2)
        t2 = self.second_transform(x)
        x = torch.matmul(x, t2).transpose(1, 2)
        x = self.second_group_of_MLPs(x)
        x = self.maxpool(x)
        return x, t2


class PointNetClassification(nn.Module):
    def __init__(self, n_classes, regulation_weight=1e-3) -> None:
        super().__init__()
        self.PointNet = PointNet()
        self.classification_head = nn.Sequential(
            MLP(1024, 512),
            nn.Dropout(.7),
            MLP(512, 256),
            nn.Dropout(.7),
            nn.Linear(256, n_classes)
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.regulation_weight = regulation_weight

    def forward(self, x):
        x, t2 = self.PointNet(x)
        x = self.classification_head(x.squeeze())
        return x, t2

    def get_loss_and_acc(self, data, label):
        predicted, transformation = self.forward(data)
        classification_loss = self.loss_fn(F.softmax(predicted, dim=-1), label)
        acc = (torch.argmax(predicted, -1) == label).sum()
        regulation_loss = torch.norm(torch.eye(64, device=transformation.device) - torch.matmul(transformation, transformation.transpose(1, 2)), p=2, dim=(1, 2)).mean()
        return classification_loss, regulation_loss, acc


if __name__ == '__main__':
    model = PointNetClassification(40)
    x = torch.rand((8, 2048, 3))
    y = torch.randint(0, 40, size=(8, ))
    l, _, acc = model.get_loss_and_acc(x, y)

