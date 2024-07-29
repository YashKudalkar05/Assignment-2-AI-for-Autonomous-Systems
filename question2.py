import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from datasets import load_dataset

# Load the MNIST dataset
mnist = load_dataset("ylecun/mnist")
train_data = mnist['train']
test_data = mnist['test']


# Convert the dataset to numpy arrays
def preprocess(data):
    images = np.array([np.array(img['image']) for img in data])
    labels = np.array([img['label'] for img in data])
    
    # Normalize images
    images = images / 255.0
    
    # Flatten images
    images = images.reshape(-1, 28 * 28)
    
    return images, labels

train_images, train_labels = preprocess(train_data)
test_images, test_labels = preprocess(test_data)


# Define the neural network
model = Sequential([
    Dense(5, activation='sigmoid', input_shape=(784,)),  # Hidden layer with 5 neurons
    Dense(10, activation='softmax')                      # Output layer with 10 neurons
])

# Compile the model
model.compile(
    optimizer=Adam(),
    loss=SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Summary of the model
model.summary()

# Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")

