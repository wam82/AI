import os

import torch
from sklearn.metrics import accuracy_score, classification_report
from torch import device
from torchvision.datasets import ImageFolder

from cnn_models import CNN, CNNVariant1, CNNVariant2
from data_loader import load_data
from utils import load_model
from torchvision import transforms

def evaluate_model_locally(model):
    # Load model
    model.eval()

    # Load the test dataset
    _, _, test_loader = load_data() # Needs a custom load data as the classes differ, they are now Young, Middle-Aged
    # and Senior.These are the classes for Age Bias. Other classes will appear for the second bias we chose
    # being race or gender. Path to Age sorted images is ../Images/Sorted_for_Bias/Age/
    # Age folder has both training and testing dataset, which are merged in the count but separate in the folders.
    # I don't know if we need to evaluate on both training and testing dataset.
    y_true = []
    y_pred = []

    # Iterate over test set and collect predictions
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

    # This is used to get the accuracy to display in Table 2.
    accuracy = accuracy_score(y_true, y_pred)

    # This is used to get the other measures in Table 2.
    # From what I understand, the measures should be calculated per class, so
    # it's just plug the results displayed
    report = classification_report(y_true, y_pred)

    return accuracy, report

# Used to count the total amount of images in each class. It works.
def count_images(root_folder, bias, categories):
    bias_folder = os.path.join(root_folder, bias)
    image_counts = {cat: 0 for cat in categories}

    for phase in ['Training', 'Testing']:
        for cat in categories:
            folder = os.path.join(bias_folder, phase, cat)
            if not os.path.isdir(folder):
                raise FileNotFoundError(f"{folder} is not a directory")
            if(os.path.exists(folder)):
                image_counts[cat] += len(os.listdir(folder))

    return image_counts

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the models
    models = {
        'Main Model': CNN().to(device),
        'Variant 1': CNNVariant1().to(device),
        'Variant 2': CNNVariant2().to(device)
    }

    root_folder = os.path.join(os.path.expanduser("~"), "Desktop", "AI", "Images", "Sorted_for_Bias")

    # Load the trained models
    models['Main Model'] = load_model(models['Main Model'], "best_main_model.pth")
    models['Variant 1'] = load_model(models['Variant 1'], "best_variant1_model.pth")
    models['Variant 2'] = load_model(models['Variant 2'], "best_variant2_model.pth")

    bias = "Age"
    class_names = ['Young', 'Middle-Aged', 'Senior']
    age_image_counts = count_images(root_folder, bias, class_names)
    for cat, count in age_image_counts.items():
        print(f"{cat}: {count}")

    # The code stops working here, I don't know how to fix error of invalid combination of arguments
    # # from passing models to evaluate_model_locally.
    for name, model in models.items():
        print(f'Evaluating {name} on the test set...')
        accuracy, report = evaluate_model_locally(model)

    # accuracy, report = evaluate_model(load_model(models['Main Model'], "best_main_model.pth"))
    # print("\nModel Evaluation:")
    # print(f"Accuracy: {accuracy:.4f}")
    # print("Classification Report:\n", report)

if __name__ == '__main__':
    main()