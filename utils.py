import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import seaborn as sns

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    test_loss = running_loss / len(test_loader.dataset)
    test_accuracy = accuracy_score(all_labels, all_preds)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    return test_loss, test_accuracy, all_preds, all_labels

def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()

def calculate_metrics(y_true, y_pred, average='macro'):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average)
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy, precision, recall, f1

def summarize_metrics(models, test_loader, criterion, device, class_names):
    results = []
    for name, model in models.items():
        _, _, y_pred, y_true = evaluate_model(model, test_loader, criterion, device)
        macro_metrics = calculate_metrics(y_true, y_pred, average='macro')
        micro_metrics = calculate_metrics(y_true, y_pred, average='micro')
        results.append({
            'Model': name,
            'Macro Accuracy': macro_metrics[0], 'Macro Precision': macro_metrics[1], 'Macro Recall': macro_metrics[2], 'Macro F1': macro_metrics[3],
            'Micro Accuracy': micro_metrics[0], 'Micro Precision': micro_metrics[1], 'Micro Recall': micro_metrics[2], 'Micro F1': micro_metrics[3],
        })
        plot_confusion_matrix(y_true, y_pred, class_names, title=f'Confusion Matrix - {name}')
    return results
