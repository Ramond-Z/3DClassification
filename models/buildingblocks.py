import torch.nn as nn


class SharedMLP(nn.Module):
    def __init__(self, in_channel, out_channel, activate=True, bn=True) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, 1),
            nn.BatchNorm1d(out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)


class MLP(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_channel, out_channel),
            nn.BatchNorm1d(out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        if len(x.shape) > 2:
            x = x.squeeze()
        return self.layers(x)
    

class SeperableConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, depth_multiplier) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(in_channel, in_channel * depth_multiplier, kernel_size, groups=in_channel),
            nn.Conv1d(in_channel * depth_multiplier, out_channel, kernel_size),
        )

    def forward(self, x):
        return self.convs(x)


class debug(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        print(x.shape)
        return x
