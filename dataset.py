from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader, random_split
import torch

def get_mnist(data_path="./data"):
    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    
    trainset = MNIST(root=data_path, train=True, 
                     download=True, transform=tr)
    testset = MNIST(root=data_path, train=False, 
                     download=True, transform=tr)
    
    return trainset, testset

def load_dataset(num_partitions, batch_size, data_path="./data", val_ratio=0.1, seed=42):
    trainset, testset = get_mnist(data_path=data_path)
    
    # Split trainset into partitions
    num_images = len(trainset) // num_partitions
    partition_len = [num_images] * num_partitions
    trainsets = random_split(trainset, partition_len, torch.Generator().manual_seed(seed))
    
    train_loaders = list()
    val_loaders = list()
    test_loader = DataLoader(testset, batch_size=batch_size * 2, shuffle=False, num_workers=2)
    
    for trainset in trainsets:
        num_val = int(len(trainset) * val_ratio)
        num_train = len(trainset) - num_val
        train, val = random_split(trainset, [num_train, num_val], torch.Generator().manual_seed(seed))
        train_loaders.append(DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2))
        val_loaders.append(DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=2))
    
    return train_loaders, val_loaders, test_loader