import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
import random

# Define the paths to the image directories
base_path = 'Images'
train_path = os.path.join(base_path, 'Training')
test_path = os.path.join(base_path, 'Testing')
classes = ['anger', 'focused', 'happy', 'neutral']

# Function to load images and labels
def load_dataset(base_path, classes, image_size=(96, 96)):
    images = []
    labels = []
    for label, class_name in enumerate(classes):
        class_path = os.path.join(base_path, class_name)
        print(f"Loading images from: {class_path}")  # Debug print
        if not os.path.exists(class_path):
            print(f"Directory does not exist: {class_path}")  # Debug print
            continue
        for filename in os.listdir(class_path):
            img_path = os.path.join(class_path, filename)
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                print(f"Skipping non-image file: {img_path}")  # Debug print
                continue
            print(f"Reading image: {img_path}")  # Debug print
            img = io.imread(img_path)
            img_resized = resize(img, image_size, anti_aliasing=True)
            images.append(img_resized)
            labels.append(label)
    return np.array(images), np.array(labels)

# Load the datasets
train_images, train_labels = load_dataset(train_path, classes)
test_images, test_labels = load_dataset(test_path, classes)

# Plot Class Distribution
def plot_class_distribution(labels, classes, title):
    plt.figure(figsize=(10, 6))
    unique, counts = np.unique(labels, return_counts=True)
    plt.bar(classes, counts, color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.title(f'Class Distribution - {title}')
    plt.show()

# Plot Pixel Intensity Distribution per Class
def plot_pixel_intensity_distribution(images, labels, classes, title):
    plt.figure(figsize=(15, 10))
    for i, class_name in enumerate(classes):
        class_images = images[labels == i]
        pixel_values = class_images.flatten()
        plt.subplot(2, 2, i+1)
        for channel, color in zip(range(3), ['red', 'green', 'blue']):
            channel_values = class_images[:, :, :, channel].flatten()
            plt.hist(channel_values, bins=50, color=color, alpha=0.5, label=f'{color}')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.title(f'Pixel Intensity Distribution - {class_name} ({title})')
        plt.legend()
    plt.tight_layout()
    plt.show()

# Display Sample Images and Histograms
def display_sample_images_and_histograms(images, labels, classes, title, samples_per_class=15):
    samples_per_row = 3
    rows = samples_per_class // samples_per_row
    for i, class_name in enumerate(classes):
        plt.figure(figsize=(15, 20))
        class_images = images[labels == i]
        # Shuffle the class images to get different samples each run
        indices = list(range(len(class_images)))
        random.shuffle(indices)
        class_images = class_images[indices]
        for j in range(samples_per_class):
            img = class_images[j]
            plt.subplot(rows, samples_per_row * 2, (j // samples_per_row) * samples_per_row * 2 + (j % samples_per_row) * 2 + 1)
            plt.imshow(img)
            plt.axis('off')
            if (j % samples_per_row) == 0:
                plt.ylabel(class_name)
            
            plt.subplot(rows, samples_per_row * 2, (j // samples_per_row) * samples_per_row * 2 + (j % samples_per_row) * 2 + 2)
            for channel, color in zip(range(3), ['red', 'green', 'blue']):
                channel_values = img[:, :, channel].flatten()
                plt.hist(channel_values, bins=50, color=color, alpha=0.5, label=f'{color}')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            plt.legend()
        plt.suptitle(f'Sample Images and Pixel Intensity Histograms from Each Class ({title}) - {class_name}')
        plt.tight_layout()
        plt.show()

# Execute the functions for training data
plot_class_distribution(train_labels, classes, 'Training Set')
plot_pixel_intensity_distribution(train_images, train_labels, classes, 'Training Set')
display_sample_images_and_histograms(train_images, train_labels, classes, 'Training Set')

# Execute the functions for testing data
plot_class_distribution(test_labels, classes, 'Testing Set')
plot_pixel_intensity_distribution(test_images, test_labels, classes, 'Testing Set')
display_sample_images_and_histograms(test_images, test_labels, classes, 'Testing Set')
