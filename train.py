from accelerate import Accelerator
from models.config import get_model
from dataset import ModelNet40, augmentation
from torch.utils.data import DataLoader
from torch.optim import AdamW
import argparse


def train(opt):
    training_set = ModelNet40(opt.root)
    training_loader = DataLoader(training_set, opt.bs, shuffle=True)
    model = get_model(opt.model)
    optimizer = AdamW(model.parameters(), opt.lr)
    accelerator = Accelerator()
    training_loader, model, optimizer = accelerator.prepare(training_loader, model, optimizer)
    if opt.ckpt is not None:
        accelerator.load_state(opt.ckpt)
    for epoch in range(opt.epochs):
        avg_loss = 0
        for batch in training_loader:
            data, label = augmentation(batch)
            loss = model.get_loss(data, label)
            avg_loss += loss.item()
            accelerator.backward(loss)
            optimizer.step()
        accelerator.log({'loss': avg_loss / len(training_loader)}, step=epoch)
    accelerator.save_state(opt.ckpt_dir)
    accelerator.end_training()


def main():
    opt = parse_args()
    train(opt)


def parse_args():
    parser = argparse.ArgumentParser(description='Training arguments')

    parser.add_argument('--root', type=str, default='data/modelnet40_ply_hdf5_2048', help='Root directory of the dataset')
    parser.add_argument('--bs', type=int, default=32, help='Batch size')
    parser.add_argument('--model', type=str, default='PointNet', help='Model architecture')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to checkpoint file')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints/', help='Directory to save checkpoints')

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    main()