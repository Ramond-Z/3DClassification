from models.Pointnet import PointNetClassification


def get_model(model_name):
    if model_name == 'PointNet':
        return PointNetClassification(40)
    elif model_name == '':
        pass
    else:
        raise NotImplementedError
