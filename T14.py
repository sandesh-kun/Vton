import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

# Define the paths to the dataset folders
dataset_folder = 'D:/datasets'
image_folder = os.path.join(dataset_folder, 'images')
parsing_folder = os.path.join(dataset_folder, 'segm')
keypoints_folder = os.path.join(dataset_folder, 'keypoints')
labels_shape_folder = os.path.join(dataset_folder, 'labels', 'shape')
labels_texture_folder = os.path.join(dataset_folder, 'labels', 'texture')

# Define the paths to other files
captions_path = os.path.join(dataset_folder, 'captions', 'captions.json')

# Define the transformations for preprocessing
transform = transforms.Compose([
    transforms.Resize((750, 1101)),  # Resize images
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images
])

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.images = os.listdir(image_folder)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_name = self.images[index]
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path)
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, torch.tensor(0)  # Assign dummy class label 0 to each image

# Load the dataset
dataset = CustomDataset(image_folder, transform=transform)

# Define the dataloader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the model architecture for clothes code segmentation
class ClothesCodeSegmentationModel(nn.Module):
    def __init__(self):
        super(ClothesCodeSegmentationModel, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(2, 2, kernel_size=3, padding=1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Create an instance of the model
model = ClothesCodeSegmentationModel()

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Function to build and train the model
def build_and_train_model(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Calculate the loss
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1} Loss: {running_loss / len(dataloader)}")
    
    print("Training complete.")

# Function to test the model
def test_model(model, dataloader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total * 100
    print(f"Accuracy: {accuracy}%")

# Function to evaluate the model
def evaluate_model(model, dataloader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.append(predicted)
    
    return torch.cat(predictions)

# Function to predict with the model
def predict(model, image):
    model.eval()
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        _, predicted = torch.max(output.data, 1)
    
    return predicted.item()

# Train the model
build_and_train_model(model, dataloader, criterion, optimizer, num_epochs=5)

# Test the model
test_model(model, dataloader)

# Save the model
torch.save(model.state_dict(), 'clothes_code_segmentation_model.pth')

# Evaluate the model
predictions = evaluate_model(model, dataloader)

# Predict with the model
image = dataset[0][0]  # Example image
prediction = predict(model, image)

# Display the prediction
print("Predicted class:", prediction)
