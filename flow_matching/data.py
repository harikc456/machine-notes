import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from flow_matching.config import FlowConfig


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)


def build_loaders(cfg: FlowConfig, num_workers: int = 4) -> tuple[DataLoader, DataLoader]:
    """Build CIFAR-100 train and val DataLoaders.

    Train: RandomCrop + HorizontalFlip + Normalize
    Val:   Normalize only
    Returns (train_loader, val_loader).
    """
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    train_set = torchvision.datasets.CIFAR100(
        root=cfg.data_root, train=True,  download=True, transform=train_transform
    )
    val_set = torchvision.datasets.CIFAR100(
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
