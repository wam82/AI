import torch
import torchvision.transforms as transforms
from PIL import Image
from cnn_models import CNN, CNNVariant1, CNNVariant2
from utils import load_model
import os

def load_image(image_path):
    # Define the transformation for the input image
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def predict(image, model, device):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

def main(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    models = {
        'Main Model': CNN().to(device),
        'Variant 1': CNNVariant1().to(device),
        'Variant 2': CNNVariant2().to(device)
    }

    # Load the trained models
    models['Main Model'] = load_model(models['Main Model'], "best_main_model.pth")
    models['Variant 1'] = load_model(models['Variant 1'], "best_variant1_model.pth")
    models['Variant 2'] = load_model(models['Variant 2'], "best_variant2_model.pth")

    # Load and preprocess the image
    image = load_image(image_path)

    # Define class names
    class_names = ['anger', 'focused', 'happy', 'neutral']

    # Predict the label for each model
    for name, model in models.items():
        predicted_class_idx = predict(image, model, device)
        predicted_class = class_names[predicted_class_idx]
        print(f'{name} predicted class: {predicted_class}')

if __name__ == "__main__":
    image_path = input("Enter the path to the image: ")
    main(image_path)
