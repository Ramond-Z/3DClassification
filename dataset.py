from torch.utils.data import Dataset, DataLoader, random_split
from torch import from_numpy

import torch
import numpy as np
import h5py
import os


class ModelNet40(Dataset):
    def __init__(self, root, split='train') -> None:
        super().__init__()
        self.root_dir = root
        self.directory = os.path.join(root, '{}_files.txt'.format(split))
        self.data, self.label = self.load_h5_files_from_directory(self.directory)

    def load_h5_files_from_directory(self, directory):
        files = [line.rstrip() for line in open(directory)]
        datas = []
        labels = []
        for file in files:
            f = h5py.File(file)
            datas.append(f['data'][:])
            labels.append(f['label'][:])
        datas = np.concatenate(datas, axis=0)
        labels = np.concatenate(labels, axis=0)
        return datas, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx], self.label[idx])


def prepare_dataset(root, validation_split=.1):
    full_dataset = ModelNet40(root)
    test_set = ModelNet40(root, 'test')
    if validation_split > 0:
        train_length = len(full_dataset) * (1 - validation_split)
        validation_length = len(full_dataset) - train_length
        train_set, validation_set = random_split(full_dataset, [train_length, validation_length])
        return train_set, validation_set, test_set
    else:
        return full_dataset, None, test_set


def rotation(batch_data):
    rotated_data = torch.zeros_like(batch_data)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = torch.asarray([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]], dtype=batch_data.dtype, device=batch_data.device)
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = torch.matmul(shape_pc, rotation_matrix)
    return rotated_data


def jitter(batch_data, sigma=.01, clip=.05):
    B, N, C = batch_data.shape
    assert (clip > 0)
    jittered_data = torch.clip(sigma * torch.randn_like(batch_data), -clip, clip)
    jittered_data += batch_data
    return jittered_data


def augmentation(batch_data):
    data, label = batch_data
    return ((jitter(rotation(data))).to(torch.float32), label.squeeze())


if __name__ == '__main__':
    data = ModelNet40('./data/modelnet40_ply_hdf5_2048')
    loader = DataLoader(data, 8, True)
    for batch in loader:
        data, label = augmentation(batch)
        print(data.shape, label.shape)
