import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def load_data():
    # Data transformations with augmentation
    transform_train = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Define the paths
    base_dir = os.path.join(os.path.expanduser("~"), "Desktop", "AI", "Images")
    train_dir = os.path.join(base_dir, "Training")
    test_dir = os.path.join(base_dir, "Testing")

    # Load datasets
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform_test)

    # Split train dataset into training and validation
    train_size = int(0.85 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return train_loader, val_loader, test_loader
