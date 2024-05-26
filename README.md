# 3D Classification

This repository contains a PyTorch implementation of several 3D classification algorithms, including PointNet and PointCNN.

## Requirements

To install the required packages, you can run:

```bash
pip install torch h5py accelerate numpy
```

or simply run:

```bash
pip install -r requirements.txt
```

## Dataset

We conducted experiments on the ModelNet40 dataset, which is available at [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip). A bash script for preparing the dataset is also provided. To prepare the dataset, please run:

```bash
sh prepare_dataset.sh
```

## Training

We utilize Huggingface's `accelerate` library for training. Notice that we **do not** support distributed training yet. To configure the training environment, please run:

```bash
accelerate config
```

After configuration, you can launch the training with:

```bash
accelerate launch train.py <args>
```

## Directory Structure

The directory structure of this repository is as follows:

```
3D-Classification/
├── data/                   # Directory for storing the dataset
├── models/                 # Directory containing model definitions
├── train.py                # Training script
├── prepare_dataset.sh      # Script to prepare the dataset
└── README.md               # This README file
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.