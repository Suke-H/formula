import torch.nn as nn
from omegaconf import DictConfig
import multiprocessing as mp
import hydra
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

class Net(nn.Sequential):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(
            nn.Conv2d(1, cfg.n_channels, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(cfg.n_channels, cfg.n_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            nn.Linear(cfg.n_channels * 2, cfg.n_hidden),
            nn.ReLU(True),
            nn.Linear(cfg.n_hidden, cfg.n_hidden // 2),
            nn.ReLU(True),
            nn.Linear(cfg.n_hidden // 2, 10),
            nn.Softmax(dim=1)
        )


@hydra.main(config_path='config', config_name='train')
def app(cfg: DictConfig) -> float:
    # train_dataset = MNIST(
    #     root=hydra.utils.get_original_cwd() + 'data',
    #     download=True,
    #     train=True,
    #     transform=transforms.Compose([lambda x: np.array(x), transforms.ToTensor()]),
    #     target_transform=lambda x: torch.as_tensor(x)
    # )
    # val_dataset = MNIST(
    #     root=hydra.utils.get_original_cwd() + 'data',
    #     download=True,
    #     train=False,
    #     transform=transforms.Compose([lambda x: np.array(x), transforms.ToTensor()]),
    #     target_transform=lambda x: torch.as_tensor(x)
    # )

    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=False, 
        transform=transforms.Compose([lambda x: np.array(x), transforms.ToTensor()]),
        target_transform=lambda x: torch.as_tensor(x)
    )

    val_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=False, 
        transform=transforms.Compose([lambda x: np.array(x), transforms.ToTensor()]),
        target_transform=lambda x: torch.as_tensor(x)
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=mp.cpu_count(),
        pin_memory=True,
        shuffle=True,
        drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        num_workers=mp.cpu_count(),
        pin_memory=True,
        drop_last=True
    )

    model = Net(cfg).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    result: float = train(model, optimizer, train_dataloader, val_dataloader, cfg.max_epochs)

    return result


if __name__ == "__main__":
    app()