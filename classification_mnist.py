# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# Import FASHION MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Preprocess the data
train_images = train_images / 255.0
test_images = test_images / 255.0

# Custom Activation Functions
# f(x) = sin(x)
def sin_act(x):
  return tf.math.sin(x, name=None)

# f(x) = ln(1 + e^x)
def softPlus(x):
  return tf.math.log(1.0 + tf.math.exp(x, name=None))

# f(x) = arctan(x)
def arcTan(x):
  return tf.math.atan(x)

# f(x) = ((x^2 + 1)^(1/2))/2 - 1
def bentIdentity(x):
  return ((tf.math.sqrt(x**2+1) - 1.0)/2.0 + x)

# f(x) = ln(1 + (e^x/(1 + e^x)))
def custom_1(x):
  return tf.math.log(1.0 + (tf.math.exp(x) / (1.0 + tf.math.exp(x))))
  
# f(x) = ((x^2 + 1)^(1/2) + x)/2 - 1
def custom_2(x):
  return ((tf.math.sqrt(x**2+1) - 1.0) / 2.0 + 0.5*x)


# Setup layers
def setupLayers(activationFunction):
  model = keras.Sequential([
      keras.layers.Flatten(input_shape=(28, 28)),
      keras.layers.Dense(128, activation=activationFunction),
      keras.layers.Dense(10)
  ])
  return model
  
# Compile Model
def compileModel(model):
  model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
  
# Train Model
def trainModel(model):
  history = model.fit(train_images, train_labels, epochs=15, validation_data=(test_images, test_labels))
  return history
  
# Evaluate Model
def evaluateModel(model):
  test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
  print('\nTest accuracy: ' + str(test_acc*100) + '%')
  
# Plot
def plot(history, title, type_1, type_2, yLabel):
  plt.plot(history.history[type_1], label=type_1)
  plt.plot(history.history[type_2], label = type_2)
  plt.title(title)
  plt.xlabel('Epoch')
  plt.ylabel(yLabel)
  plt.ylim([0.5, 1])
  plt.legend(loc='lower right')

# Run Model
def run(activationFunction):
  model = setupLayers(activationFunction)
  compileModel(model)
  print('***************************** '+ str(activationFunction) + 'Activation ***********************')
  history = trainModel(model)
  evaluateModel(model)
  return history

# Using Sigmoid Activation
sigmoidHistory = run('sigmoid')
plot(sigmoidHistory, 'Sigmoid Activation: MNIST Dataset', 'accuracy', 'val_accuracy', 'Accuracy')

# Using ReLU Activation
reluHistory = run('relu')
plot(reluHistory, 'ReLU Activation: MNIST Dataset', 'accuracy', 'val_accuracy', 'Accuracy')

# Using Tanh Activation
tanhHistory = run('tanh')
plot(tanhHistory, 'tanh Activation: MNIST Dataset', 'accuracy', 'val_accuracy', 'Accuracy')

# Using arcTan Activation
arcTanHistory = run(arcTan)
plot(arcTanHistory, 'arcTan Activation: MNIST Dataset', 'accuracy', 'val_accuracy', 'Accuracy')

# Using Custom-1 Activation
custom_1_History = run(custom_1)
plot(custom_1_History, 'Custom-1 Activation: MNIST Dataset', 'accuracy', 'val_accuracy', 'Accuracy')

# Using Custom-2 Activation
custom_2_History = run(custom_2)
plot(custom_2_History, 'Custom-2 Activation: MNIST Dataset', 'accuracy', 'val_accuracy', 'Accuracy')

# Validation Accuracy Comparison
plt.plot(sigmoidHistory.history['val_accuracy'], label = 'Sigmoid')
plt.plot(reluHistory.history['val_accuracy'], label='ReLU')
plt.plot(tanhHistory.history['val_accuracy'], label='tanh')
plt.plot(arcTanHistory.history['val_accuracy'], label='arcTan')
plt.plot(custom_1_History.history['val_accuracy'], label='Custom-1')
plt.plot(custom_2_History.history['val_accuracy'], label='Custom-2')
plt.title('Activations: MNIST Dataset')
plt.xlabel('Epoch')
plt.ylabel('Val_Accuracy')
plt.ylim([0.8, 0.9])
plt.legend(loc='lower right')

# Accuracy Comparison
plt.plot(sigmoidHistory.history['accuracy'], label = 'Sigmoid')
plt.plot(reluHistory.history['accuracy'], label='ReLU')
plt.plot(tanhHistory.history['accuracy'], label='tanh')
plt.plot(arcTanHistory.history['accuracy'], label='arcTan')
plt.plot(custom_1_History.history['accuracy'], label='Custom-1')
plt.plot(custom_2_History.history['accuracy'], label='Custom-2')
plt.title('Activations: MNIST Dataset')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.8, 0.95])
plt.legend(loc='lower right')

# Validation Loss Comparison
plt.plot(sigmoidHistory.history['val_loss'], label = 'Sigmoid')
plt.plot(reluHistory.history['val_loss'], label='ReLU')
plt.plot(tanhHistory.history['val_loss'], label='tanh')
plt.plot(arcTanHistory.history['val_loss'], label='arcTan')
plt.plot(custom_1_History.history['val_loss'], label='Custom-1')
plt.plot(custom_2_History.history['val_loss'], label='Custom-2')
plt.title('Activations: MNIST Dataset')
plt.xlabel('Epoch')
plt.ylabel('Val_Loss')
plt.ylim([0.3, 0.5])
plt.legend(loc='lower right')

# Loss Comparison
plt.plot(sigmoidHistory.history['loss'], label = 'Sigmoid')
plt.plot(reluHistory.history['loss'], label='ReLU')
plt.plot(tanhHistory.history['loss'], label='tanh')
plt.plot(arcTanHistory.history['loss'], label='arcTan')
plt.plot(custom_1_History.history['loss'], label='Custom-1')
plt.plot(custom_2_History.history['loss'], label='Custom-2')
plt.title('Activations: MNIST Dataset')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 0.6])
plt.legend(loc='lower right')
