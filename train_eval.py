import torch
import torch.optim as optim
from cnn_models import CNN, CNNVariant1, CNNVariant2
from data_loader import load_data
from utils import save_model, evaluate_model
import os

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(train_loader.dataset)

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
    val_loss = running_loss / len(val_loader.dataset)
    val_accuracy = correct.double() / len(val_loader.dataset)
    return val_loss, val_accuracy

def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=25, patience=10):
    best_loss = float('inf')
    best_model_wts = model.state_dict()
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print('Early stopping!')
                break

    model.load_state_dict(best_model_wts)
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, val_loader, test_loader = load_data()

    # Initialize the models, criterion, and optimizer
    models = {
        'Main Model': CNN().to(device),
        'Variant 1': CNNVariant1().to(device),
        'Variant 2': CNNVariant2().to(device)
    }
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(models['Main Model'].parameters(), lr=0.001)

    # Train and evaluate the main model
    print("Training Main Model...")
    models['Main Model'] = train_and_evaluate(models['Main Model'], train_loader, val_loader, criterion, optimizer, device, num_epochs=2, patience=10)
    save_model(models['Main Model'], "best_main_model.pth")

    print("Training Variant 1...")
    optimizer = optim.Adam(models['Variant 1'].parameters(), lr=0.001)
    models['Variant 1'] = train_and_evaluate(models['Variant 1'], train_loader, val_loader, criterion, optimizer, device, num_epochs=2, patience=10)
    save_model(models['Variant 1'], "best_variant1_model.pth")

    print("Training Variant 2...")
    optimizer = optim.Adam(models['Variant 2'].parameters(), lr=0.001)
    models['Variant 2'] = train_and_evaluate(models['Variant 2'], train_loader, val_loader, criterion, optimizer, device, num_epochs=2, patience=10)
    save_model(models['Variant 2'], "best_variant2_model.pth")

    # Evaluate the models on the test set
    for name, model in models.items():
        print(f'Evaluating {name} on the test set...')
        evaluate_model(model, test_loader, criterion, device)

if __name__ == "__main__":
    main()
