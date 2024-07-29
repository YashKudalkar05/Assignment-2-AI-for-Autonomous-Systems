import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from datasets import load_dataset

# Load the MNIST dataset
mnist = load_dataset("ylecun/mnist")
train_data = mnist['train']
test_data = mnist['test']

# Convert the dataset to numpy arrays and preprocess
def preprocess(data):
    images = np.array([np.array(img['image']) for img in data])
    labels = np.array([img['label'] for img in data])

    images = images / 255.0

    images = images.reshape(-1, 28 * 28)
    
    return images, labels

train_images, train_labels = preprocess(train_data)
test_images, test_labels = preprocess(test_data)

# Define the fully connected neural network (MLP)
mlp_model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),  
    Dense(64, activation='relu'),                   
    Dense(10, activation='softmax')                     
])

# Compile the model
mlp_model.compile(
    optimizer=Adam(),
    loss=SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Summary of the model
mlp_model.summary()

# Train the model
mlp_model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = mlp_model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")

################ Part 2 ###################


# Load the MNIST dataset
mnist = load_dataset("ylecun/mnist")
train_data = mnist['train']
test_data = mnist['test']

# Convert the dataset to numpy arrays and preprocess
def preprocess(data):
    images = np.array([np.array(img['image']) for img in data])
    labels = np.array([img['label'] for img in data])

    images = images / 255.0
    
    images = images.reshape(-1, 28, 28, 1)
    
    return images, labels

train_images, train_labels = preprocess(train_data)
test_images, test_labels = preprocess(test_data)

# Define the CNN model
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),                               
    Conv2D(64, (3, 3), activation='relu'),                       
    MaxPooling2D(pool_size=(2, 2)),                             
    Flatten(),                                              
    Dense(128, activation='relu'),                          
    Dense(10, activation='softmax')                         
])

# Compile the model
cnn_model.compile(
    optimizer=Adam(),
    loss=SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Summary of the model
cnn_model.summary()

# Train the model
cnn_model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = cnn_model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")
