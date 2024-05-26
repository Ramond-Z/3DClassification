from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from models.config import get_model
from dataset import ModelNet40, augmentation
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from datetime import datetime
import argparse


def train(opt):
    training_set = ModelNet40(opt.root)
    training_loader = DataLoader(training_set, opt.bs, shuffle=True)
    model = get_model(opt.model)
    optimizer = AdamW(model.parameters(), opt.lr)
    scheduler = StepLR(optimizer, 20, .5)
    project = ProjectConfiguration('.', './log')
    accelerator = Accelerator(log_with='tensorboard', project_config=project)
    accelerator.init_trackers(opt.experience_name)
    training_loader, model, optimizer, scheduler = accelerator.prepare(training_loader, model, optimizer, scheduler)
    if opt.ckpt is not None:
        accelerator.load_state(opt.ckpt)
    for epoch in range(opt.epochs):
        avg_loss = 0
        avg_classification_loss = 0
        avg_regulation_loss = 0
        total_acc = 0
        for batch in training_loader:
            optimizer.zero_grad()
            data, label = augmentation(batch)
            classification_loss, regulation_loss, acc = model.get_loss_and_acc(data, label)
            loss = classification_loss + opt.reg_weight * regulation_loss
            avg_loss += loss.item()
            avg_classification_loss += classification_loss.item()
            avg_regulation_loss += regulation_loss.item()
            total_acc += acc
            accelerator.backward(loss)
            optimizer.step()
        scheduler.step()
        accelerator.log({'loss': avg_loss / len(training_loader),'cls_loss': avg_classification_loss / len(training_loader), 'reg_loss': regulation_loss / len(training_loader), 'acc': total_acc / len(training_set)}, step=epoch)
        print('epoch: {}, loss: {}, acc: {}'.format(epoch, avg_loss / len(training_loader), total_acc / len(training_set)))
    accelerator.save_state('{}/{}'.format(opt.ckpt_dir, opt.experience_name))
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
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to checkpoint file')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints/', help='Directory to save checkpoints')
    parser.add_argument('--experience_name', type=str, default='experiment-{}'.format(datetime.now()))
    parser.add_argument('--reg_weight', type=float, default=1e-3)

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    main()