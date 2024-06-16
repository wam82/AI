import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def load_data():
    # Data transformations with augmentation for training
    transform_train = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Data transformations for validation and testing
    transform_test = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Define the paths
    base_dir = os.path.join(os.path.expanduser("~"), "Desktop", "AI", "Images")
    train_dir = os.path.join(base_dir, "Training")
    test_dir = os.path.join(base_dir, "Testing")

    # Debug prints to verify paths
    print(f"Base directory: {base_dir}")
    print(f"Training directory: {train_dir}")
    print(f"Testing directory: {test_dir}")

    # Check if directories exist
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Testing directory not found: {test_dir}")

    # Load datasets
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform_test)

    # Debug prints to verify dataset contents
    print(f"Training dataset classes: {train_dataset.classes}")
    print(f"Testing dataset classes: {test_dataset.classes}")

    # Split train dataset into training and validation
    train_size = int(0.85 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader
