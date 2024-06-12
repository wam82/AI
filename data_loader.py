import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def load_data():
    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # Define the paths
    base_dir = os.path.join(os.path.expanduser("~"), "Desktop", "AI", "Images")
    train_dir = os.path.join(base_dir, "Training")
    test_dir = os.path.join(base_dir, "Testing")

    # Load datasets
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    # Split train dataset into training and validation
    train_size = int(0.85 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return train_loader, val_loader, test_loader
