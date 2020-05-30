# Import Tensorflow & other utilities
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


# Download & prepare CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


# Custom activation functions
def sin_act(x):
  return tf.math.sin(x, name=None)

def softPlus(x):
  return tf.math.log(1.0 + tf.math.exp(x, name=None))

def arcTan(x):
  return tf.math.atan(x)

def bentIdentity(x):
  return ((tf.math.sqrt(x**2+1) - 1.0)/2.0 + x)

def custom_1(x):
  return tf.math.log(1.0 + (tf.math.exp(x) / (1.0 + tf.math.exp(x))))

def custom_2(x):
  return ((tf.math.sqrt(x**2+1) - 1.0) / 2.0 + 0.5*x)

def custom_3(x):
  return ((tf.math.sqrt(x**2+1) - 1.0) / 2.0 + x*(1.0/(1.0+tf.math.exp(0.5)) + 0.5))

# Setup CNN
def createConvolutionalBase(activationFunction):
  model = models.Sequential()
  model.add(layers.Conv2D(32, (3, 3), activation=activationFunction, input_shape=(32, 32, 3)))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation=activationFunction))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation=activationFunction))
  
  def addDenseLayerOnTop(model, activationFunction):
  model.add(layers.Flatten())
  model.add(layers.Dense(64, activation=activationFunction))
  model.add(layers.Dense(10))
  return model
def custom_4(x):
  return (tf.math.log((1.0+ tf.math.sin(x))/2.0))
  

# Compile & train model
def compileAndTrainModel(model, epochs):
  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  history = model.fit(train_images, train_labels, epochs=epochs, 
                      validation_data=(test_images, test_labels))
  return history

# evaluate Model
def evaluateModel(model, history):
  test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
  return test_loss, test_acc

# plot
def plot(history, title, type_1, type_2, yLabel):
  plt.plot(history.history[type_1], label=type_1)
  plt.plot(history.history[type_2], label = type_2)
  plt.title(title)
  plt.xlabel('Epoch')
  plt.ylabel(yLabel)
  plt.ylim([0.3, 0.9])
  plt.legend(loc='lower right')


# Run Model
def run(activationFunction, epochs):
  print('***************************'+ str(activationFunction) + ' ACTIVATION FUNCTION**********************')
  model = createConvolutionalBase(activationFunction)

  addDenseLayerOnTop(model, activationFunction)

  history = compileAndTrainModel(model, epochs)
  test_loss, test_acc = evaluateModel(model, history)

  print('Test Accuracy -> '+str((test_acc)*100)+'%')
  print('Test Loss -> '+str(test_loss)+'%')
  
  return history
  


# Using Sigmoid Activation
sigmoidHistory = run('sigmoid', 10)
plot(sigmoidHistory, 'Sigmoid Activation: CIFAR-10 Dataset', 'accuracy', 'val_accuracy', 'Accuracy')

# Using ReLU Activation
reluHistory = run('relu', 10)
plot(reluHistory, 'ReLU Activation: CIFAR-10 Dataset', 'accuracy', 'val_accuracy', 'Accuracy')

# Using Tanh Activation
tanhHistory = run('tanh', 10)
plot(tanhHistory, 'tanh Activation: CIFAR-10 Dataset', 'accuracy', 'val_accuracy', 'Accuracy')

# Using arcTan Activation
arcTanHistory = run(arcTan, 10)
plot(arcTanHistory, 'arcTan Activation: CIFAR-10 Dataset', 'accuracy', 'val_accuracy', 'Accuracy')

# Using Custom-1 Activation
custom_1_History = run(custom_1, 10)
plot(custom_1_History, 'Custom-1 Activation: CIFAR-10 Dataset', 'accuracy', 'val_accuracy', 'Accuracy')

# Using Custom-2 Activation
custom_2_History = run(custom_2, 10)
plot(custom_2_History, 'Custom-2 Activation: CIFAR-10 Dataset', 'accuracy', 'val_accuracy', 'Accuracy')

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
