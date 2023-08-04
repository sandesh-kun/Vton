import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Set the root directory
root_dir = r"C:\Users\Sandesh\PycharmProjects\FirstProject\HE"

# Download the model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Load the image using a file dialog
Tk().withdraw()  # Hide the Tkinter root window
image_path = askopenfilename()  # Show the file dialog to select an image
img = cv2.imread(image_path)

# Define the transform to apply to the image
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Apply the transform to the image
img_tensor = transform(img)

# Pass the image through the model to get the predicted boxes and labels
model.eval()
with torch.no_grad():
    outputs = model([img_tensor])

# Convert the image to a numpy array
img = np.array(img)

# Visualize the predicted boxes on the image
boxes = outputs[0]['boxes'].cpu().numpy().astype(np.int32)
scores = outputs[0]['scores'].cpu().numpy()
labels = outputs[0]['labels'].cpu().numpy()
for box, score, label in zip(boxes, scores, labels):
    if score > 0.5:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(img, str(label), (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Display the image
plt.imshow(img)
plt.show()
