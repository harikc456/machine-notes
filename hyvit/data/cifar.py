import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


def build_loaders(cfg, num_workers: int = 4):
    """
    Build CIFAR-10 train and validation DataLoaders.

    Train augmentations: RandomCrop(32, pad=4) + HorizontalFlip + ColorJitter + Normalize
    Val augmentations:   Normalize only
    """
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=cfg.data_root, train=True,  download=True, transform=train_transform
    )
    val_set = torchvision.datasets.CIFAR10(
        root=cfg.data_root, train=False, download=True, transform=val_transform
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
