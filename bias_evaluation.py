import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from cnn_models import CNN, CNNVariant1, CNNVariant2
from data_loader import load_data
from utils import load_model, evaluate_model, save_model, plot_confusion_matrix  # Import plot_confusion_matrix
from train_eval import train_and_evaluate, validate  # Import the functions
from sklearn.model_selection import KFold
import pandas as pd

def evaluate_model_locally(model, test_loader, device, class_names):
    model.eval()
    y_true = []
    y_pred = []

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Plot the confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names, "Confusion Matrix")
    
    return accuracy, report

def count_images(root_folder, bias, categories):
    bias_folder = os.path.join(root_folder, bias)
    image_counts = {cat: 0 for cat in categories}

    for phase in ['Training', 'Testing']:
        for cat in categories:
            folder = os.path.join(bias_folder, phase, cat)
            if not os.path.isdir(folder):
                raise FileNotFoundError(f"{folder} is not a directory")
            if os.path.exists(folder):
                image_counts[cat] += len(os.listdir(folder))

    return image_counts

def k_fold_cross_validation(models, dataset, k=10):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results = {name: [] for name in models.keys()}

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f'Fold {fold + 1}/{k}')

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

        for name, model in models.items():
            model.to(device)
            model.train()

            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            model = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=2, patience=10)
            val_loss, val_accuracy = validate(model, val_loader, criterion, device)
            results[name].append({'val_loss': val_loss, 'val_accuracy': val_accuracy})

    return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = {
        'Main Model': CNN().to(device),
        'Variant 1': CNNVariant1().to(device),
        'Variant 2': CNNVariant2().to(device)
    }

    root_folder = os.path.join(os.path.expanduser("~"), "Desktop", "COMP472", "AI", "Images", "Sorted_for_Bias")

    models['Main Model'] = load_model(models['Main Model'], "best_main_model.pth")
    models['Variant 1'] = load_model(models['Variant 1'], "best_variant1_model.pth")
    models['Variant 2'] = load_model(models['Variant 2'], "best_variant2_model.pth")

    biases = {
        "Age": ['Young', 'Middle-Age', 'Senior'],
        "Gender": ['Female', 'Male']
    }

    for bias, class_names in biases.items():
        print(f"Evaluating for {bias} bias...")
        image_counts = count_images(root_folder, bias, class_names)
        for cat, count in image_counts.items():
            print(f"{cat}: {count}")

        # Load data
        train_loader, val_loader, test_loader = load_data()

        # k-Fold Cross-Validation
        print("Starting k-Fold Cross-Validation...")
        results = k_fold_cross_validation(models, train_loader.dataset, k=10)

        for name, res in results.items():
            print(f'Results for {name}:')
            for fold, metrics in enumerate(res):
                print(f'  Fold {fold + 1}: Loss={metrics["val_loss"]:.4f}, Accuracy={metrics["val_accuracy"]:.4f}')
            avg_loss = np.mean([x['val_loss'] for x in res])
            avg_accuracy = np.mean([x['val_accuracy'] for x in res])
            print(f'  Average: Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.4f}')

        # Bias Evaluation
        for name, model in models.items():
            print(f'Evaluating {name} on the test set...')
            accuracy, report = evaluate_model_locally(model, test_loader, device, class_names)
            print(f"Accuracy: {accuracy:.4f}")
            print("Classification Report:\n", report)

if __name__ == '__main__':
    main()
