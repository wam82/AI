import torch
from cnn_models import CNN, CNNVariant1, CNNVariant2
from data_loader import load_data
from utils import load_model, evaluate_model, plot_detailed_binary_confusion_matrix

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    _, _, test_loader = load_data()

    # Initialize the models
    models = {
        'Main Model': CNN().to(device),
        'Variant 1': CNNVariant1().to(device),
        'Variant 2': CNNVariant2().to(device)
    }

    # Load the trained models
    models['Main Model'] = load_model(models['Main Model'], "best_main_model.pth")
    models['Variant 1'] = load_model(models['Variant 1'], "best_variant1_model.pth")
    models['Variant 2'] = load_model(models['Variant 2'], "best_variant2_model.pth")

    class_names = ['anger', 'focused', 'happy', 'neutral']

    for name, model in models.items():
        print(f'Evaluating {name} on the test set...')
        test_loss, test_accuracy, all_preds, all_labels = evaluate_model(model, test_loader, torch.nn.CrossEntropyLoss(), device)
        print(f'{name} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
        for idx, class_name in enumerate(class_names):
            plot_detailed_binary_confusion_matrix(all_labels, all_preds, idx, class_name, title=f'Binary Confusion Matrix - {name}')

if __name__ == "__main__":
    main()
