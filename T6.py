import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import torchvision
import torchvision.models as models


# Load pre-trained model
model = models.segmentation.fcn_resnet101(pretrained=True).eval()

# Define label to class mappings
label_mappings = {
    1: 'upper_body_clothes',
    2: 'lower_body_clothes',
    3: 'full_body_clothes',
    4: 'onepiece',
    5: 'shoe',
    6: 'bag',
    7: 'accessory',
    8: 'sunglasses',
    9: 'eyeglasses',
    10: 'hat'
}

target_label = 15


def semantic_segmentation(image_path, model, target_label):
    # Load image and transform for model input
    image = Image.open(image_path).convert('RGB')
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    input_tensor = transform(image).unsqueeze(0)

    # Run image through model to get predicted mask
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
        output_predictions = output.argmax(0).byte().cpu().numpy()

    # Find the indices of all pixels with the target class label
    indices = np.where(output_predictions == target_label)

    # Extract the x and y coordinates of each pixel with the target label
    x_coords = indices[0]
    y_coords = indices[1]

    # Print the coordinates of each pixel with the target label
    for x, y in zip(x_coords, y_coords):
        print(f"({x}, {y})")
    
    output = output_predictions
    return output


def detect_clothes(mask):
    # Map labels to classes and count number of pixels for each class
    classes, counts = np.unique(mask, return_counts=True)
    classes = [label_mappings[label] for label in classes if label in label_mappings]
    counts = dict(zip(classes, counts))

    # Return dictionary of class counts
    return counts


def select_image():
    # Open file dialog to select image file
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()

    # If file is selected, perform semantic segmentation and display results
    if file_path:
        # Perform semantic segmentation
        mask = semantic_segmentation(file_path, model, target_label)

        # Detect clothes in mask
        clothes_counts = detect_clothes(mask)

        # Display input image
        input_image = plt.imread(file_path)
        plt.subplot(1, 3, 1)
        plt.imshow(input_image)
        plt.title('Input Image')

        # Display semantic segmentation mask
        plt.subplot(1, 3, 2)
        plt.imshow(mask)
        plt.title('Segmentation Mask')

        # Display class counts
        plt.subplot(1, 3, 3)
        plt.bar(clothes_counts.keys(), clothes_counts.values())
        plt.title('Clothes Counts')
        plt.xticks(rotation=90)

        # Show the plot
        plt.show()


# Call select_image function when button is clicked
tk.Button(text="Select Image", command=select_image).pack()

# Start the GUI loop
tk.mainloop()
