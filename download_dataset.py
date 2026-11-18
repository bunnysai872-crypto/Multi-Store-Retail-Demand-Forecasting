
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

print("Downloading Fashion-MNIST dataset...")

# Load dataset from Keras
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print("\nDataset downloaded successfully!")

print("Training images:", x_train.shape)
print("Training labels:", y_train.shape)

print("Testing images:", x_test.shape)
print("Testing labels:", y_test.shape)

print("\nNumber of classes:", len(set(y_train)))
print("Image dimensions:", x_train.shape[1:])
