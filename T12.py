import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the trained model
model = keras.models.load_model('model.h5')

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt',
               'Sneaker', 'Bag', 'Ankle boot']

# Load and preprocess the custom image
image_path = 'shirt.jpg'
image = Image.open(image_path).convert('L')  # Convert to grayscale if needed
image = image.resize((28, 28))  # Resize the image to match the input size of the model
image = np.array(image) / 255.0  # Normalize the pixel values

# Make predictions
predictions = model.predict(np.expand_dims(image, axis=0))
predicted_label = np.argmax(predictions[0])
predicted_class = class_names[predicted_label]

# Display the custom image and predicted label
plt.imshow(image, cmap=plt.cm.binary)
plt.axis('off')
plt.title(predicted_class)
plt.show()
