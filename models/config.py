from models.Pointnet import PointNetClassification
from models.PointCNN import PointCNNClassification


def get_model(model_name):
    if model_name == 'PointNet':
        return PointNetClassification(40)
    elif model_name == 'PointCNN':
        return PointCNNClassification(40)
    else:
        raise NotImplementedError
