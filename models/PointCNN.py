from models.buildingblocks import MLP, SharedMLP
# from buildingblocks import *


import torch.nn as nn
import torch.nn.functional as F
import torch


class X_conv(nn.Module):
    def __init__(self, num_neighbours, in_feature, out_feature, dim=3, lifting_feature=None) -> None:
        super().__init__()
        self.num_neighbours = num_neighbours
        self.dim = dim
        lifting_feature = in_feature // 4 if lifting_feature is None else lifting_feature
        self.lifting_network = SharedMLP(dim, lifting_feature)
        self.transformation_network = nn.Sequential(
            MLP(dim * num_neighbours, num_neighbours * num_neighbours),
            MLP(num_neighbours * num_neighbours, num_neighbours * num_neighbours),
            nn.Linear(num_neighbours * num_neighbours, num_neighbours * num_neighbours),
            nn.BatchNorm1d(num_neighbours * num_neighbours),
        )
        # self.conv = SeperableConv(in_feature + lifting_feature, out_feature, kernel_size, int(out_feature / (in_feature + lifting_feature)))
        self.conv = nn.Conv1d(in_feature + lifting_feature, out_feature, num_neighbours)

    def forward(self, points: torch.Tensor, features, representitives: torch.Tensor):
        b, n, c = representitives.shape
        points = points.flatten(0, 1)
        representitives = representitives.flatten(0, 1)
        points -= representitives.unsqueeze(-2)
        lifted_features = self.lifting_network(points.transpose(1, 2)).transpose(1, 2)
        if features is not None:
            features = features.flatten(0, 1)
            features = torch.concatenate([lifted_features, features], dim=-1)
        else:
            features = lifted_features
        transformation = self.transformation_network(points.flatten(1).unsqueeze(-1)).reshape(-1, self.num_neighbours, self.num_neighbours)
        feature_map = torch.bmm(transformation, features).transpose(1, 2)
        features = self.conv(feature_map)
        return representitives.reshape(b, n, self.dim), features.reshape(b, n, -1)


class PointCNNLayer(nn.Module):
    def __init__(self, in_feature, out_feature, num_neighbours, num_representitives, dilation, dim=3, lifting_feature=None) -> None:
        super().__init__()
        self.conv = X_conv(num_neighbours, in_feature, out_feature, lifting_feature=lifting_feature, dim=dim)
        self.dilation = dilation
        self.num_neighbours = num_neighbours
        self.num_representitives = num_representitives

    def forward(self, points, features):
        b, n, c = points.shape
        nn_idx, _ = knn(points, self.dilation * self.num_neighbours)
        if self.dilation > 1:
            nn_idx = nn_idx[:, :, torch.randperm(self.dilation * self.num_neighbours)[:self.num_neighbours]]
        rep_idx = torch.randperm(n)[:self.num_representitives]
        representitive_pos = points[:, rep_idx]
        neighbour_idx = nn_idx[:, rep_idx, :]
        batch_indices = torch.arange(b).view(b, 1, 1).expand(-1, self.num_representitives, self.num_neighbours)
        neighbour_pos = points[batch_indices, neighbour_idx]
        neighbour_features = features[batch_indices, neighbour_idx] if features is not None else None
        representitives, new_features = self.conv(neighbour_pos, neighbour_features, representitive_pos)
        return representitive_pos, new_features


class PointCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            PointCNNLayer(0, 48, 8, 1024, 1, lifting_feature=16),
            PointCNNLayer(48, 96, 12, 384, 2),
            PointCNNLayer(96, 192, 16, 128, 2),
            PointCNNLayer(192, 384, 16, 128, 3)
        ])

    def forward(self, points):
        features = None
        for layer in self.layers:
            points, features = layer(points, features)
        return points, features


class PointCNNClassification(nn.Module):
    def __init__(self, n_category) -> None:
        super().__init__()
        self.backbone = PointCNN()
        self.classification_head = nn.Sequential(
            SharedMLP(384, 256),
            SharedMLP(256, 128),
            nn.Dropout(),
            nn.Conv1d(128, n_category, 1),
        )
        self.avg_pooling = nn.AdaptiveAvgPool1d(n_category)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, points):
        _, features = self.backbone(points)
        logits = torch.mean(self.classification_head(features.transpose(1, 2)), -1)
        return logits

    def get_loss_and_acc(self, points, label):
        logits = self.forward(points)
        loss = self.loss_fn(F.softmax(logits, dim=-1), label)
        acc = (torch.argmax(logits, -1) == label).sum()
        return loss, torch.asarray([0], device=loss.device), acc


def knn(point_cloud, k, include_oneself=False):
    if len(point_cloud.shape) == 2:
        point_cloud = point_cloud.unsqueeze(0)
    dist_matrix = torch.cdist(point_cloud, point_cloud)

    knn_dist, knn_indices = torch.topk(dist_matrix, k=k+1, largest=False)
    if not include_oneself:
        knn_indices = knn_indices[:,: , 1:]
        knn_dist = knn_dist[:,: ,1:]
    else:
        knn_indices = knn_indices[:,: , :-1]
        knn_dist = knn_dist[:,: ,:-1]

    return knn_indices, knn_dist


if __name__ == '__main__':
    ptcnn = PointCNNClassification(40)
    pc = torch.rand((2, 1024, 3))
    labels = torch.randint(0, 40, (2, ))
    loss, _, acc = ptcnn.get_loss_and_acc(pc, labels)
