---
title: "Keras Dense Network with MNIST/CIFAR10"
teaching: 20
exercises: 0
questions:
- "Image Classification with MNIST and CIFAR10 data"
objectives:
- "MNIST, CIFAR10, Dense Layer"
keypoints:
- "Classification training, MNIST, CIFAR10, keras"
---
## Using Keras to solve a Image Classification using MNIST data

### Prepare the data

The MNIST dataset - Modified National Institute of Standards and Technology dataset

![image](https://user-images.githubusercontent.com/43855029/192849736-0d6849c2-1882-4590-a817-73e0e350ca65.png)


The MNIST database of handwritten digits from 0 to 9, has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image of 28 x 28 pixel.

The MNIST database of handwritten digits problem is considered the "Hello World" of deep learning: to correctly classify hand-written digits.

### Objectives

* Understand how deep learning can solve problems traditional programming methods cannot
* Learn about the [MNSIT handwritten digits dataset](http://yann.lecun.com/exdb/mnist/)
* Use the [Keras API](https://keras.io/) to load the MNIST dataset and prepare it for training
* Create a simple neural network to perform image classification
* Train the neural network 
* Observe the performance of the trained neural network

### Load MNIST data

```python
from tensorflow.keras.datasets import mnist
```

### Split to Training & Testing

```python
(X_train,y_train),(X_test,y_test) = mnist.load_data()
```

### Render the image using Matplotlib

```python
# Lets take a look at some images 
import matplotlib.pyplot as plt
for i in range(64):
    ax = plt.subplot(8, 8, i+1)
    ax.axis('off')
    plt.imshow(X_train[i], cmap='Greys')
```

![image](https://user-images.githubusercontent.com/43855029/192866517-53ec1bd6-ddc9-4804-84aa-97b1ecfe592b.png)


### Prepare data

In deep learning, it is common that data needs to be transformed to be in the ideal state for training. For this particular image classification problem, there are 3 tasks we should perform with the data in preparation for training:

1. Flatten the image data, to simplify the image input into the model
2. Normalize the image data, to make the image input values easier to work with for the model
3. Categorize the labels, to make the label values easier to work with for the model

#### 1. Flatten the image data

We are going to reshape each 24 bit RGB image into a single array of 28*28 = 784 continous pixels (flattening process)

![image](https://user-images.githubusercontent.com/43855029/192850134-d35fa88e-cc20-4490-9f00-6391d8ab401a.png)

```python
print(X_train.shape)
print(X_test.shape)
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2])
X_test  = X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2])
print(X_train.shape)
print(X_test.shape)
```
#### 2. Normalize the image data

```python
print(X_train.max())
print(X_test.max())
X_train = X_train/X_train.max()
X_test = X_test/X_test.max()
```

#### 3. Categorical Encoding
For all kinds of classification it's always better to classify output into categorical encoding, so the number classification would be the same as image of other type classification (car, animal, machine)


Here we utilize the Keras's utility [categorically encode values](https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical) named **utils.to_categorical**

```python
import tensorflow.keras as keras
num_categories = 10 # Ranges from 0-9

y_train = keras.utils.to_categorical(y_train,num_categories)
y_test = keras.utils.to_categorical(y_test,num_categories)
```

### Create Keras Sequential model with Dense Layers

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
```

### Create 2 hidden layers with 512 and 512 hidden nodes each and 10 output layer with Keras:

```python
# Create a Sequential model
model = Sequential()
# Create a first hidden layer, the input for the first hidden layer is input layer which has 3 variables:
model.add(Dense(units=32,activation='relu',input_shape=(784,)))
# Create a second hidden layer
model.add(Dense(units=16,activation='relu'))
# Create an output layer with only 10 variables:
model.add(Dense(units = num_categories,activation='softmax'))
```

### Summarizing the model

```python
model.summary()
```

![image](https://user-images.githubusercontent.com/43855029/192866597-330035c1-b2a8-401d-b1ea-f07144fe33f7.png)


### Compile model

```python
model.compile(optimizer="sgd",loss='categorical_crossentropy',metrics=['accuracy'])
```

![image](https://user-images.githubusercontent.com/43855029/192866661-eda717fa-0af2-4c3c-a979-c672d62ba30d.png)

### Train model

```python
history = model.fit(X_train,y_train,epochs=10,verbose=1,validation_data=(X_test,y_test))
y_predict = model.predict(X_test)

from matplotlib import pyplot as plt

def plot_acc_loss(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='best')

    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')

    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='best')

    plt.show()

plot_acc_loss(history)
```

![image](https://user-images.githubusercontent.com/43855029/192866744-255702d7-91a7-44d3-b625-211e1b18a21e.png)


### Evaluate on test data

```python
# Lets take a look at some images 
import matplotlib.pyplot as plt
import numpy as np
for i in range(16):
    ax = plt.subplot(4, 4, i+1)
    ax.axis('off')
    plt.imshow(X_test[i].reshape(28,28), cmap='Greys')
    plt.title(["guess ", np.argmax(y_predict[i])])
```

![image](https://user-images.githubusercontent.com/43855029/192866799-48619765-900f-4130-a36e-45f4da67fe43.png)




### Hyperparameter optimization


The model can be improved by adjusting several parameters in the models:

- Number of Epochs
- Hidden Layers (Number of Layers)
- Hidden Units in a layer (Width of each layer)
- Activations Functions
- Learning Rate

### Discussions:
- The Dense (regular NN) works very well with MNIST data with high accuracy (more than 0.95 for both training and testing) and small losses.
- The training speed at 558 us/step is also lightning fast
- Will this Dense model continue to work well with other 2D dataset like CIFAR10? We will answer this question in the Exercise part below!

## Exercise: Using Keras to solve a Image Classification using CIFAR10 data

### CIFAR10 database

- The CIFAR10 database consisting 60,000 color images with 10 different classes
- Each image has 32 x 32 pixels with **color** range from 0-255
- It is good database for pattern recognition and image classification task (the entire data is clean and ready for use).
- The dataset was divided into 50,000 images for training and 10,000 images for testing
- The 10 classes are **airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck**

Comparison to MNIST?

- CIFAR10 is similar to MNIST with as they have similar number of samples data (60k vs 70k for MNIST)
- CIFAR10 has **colors** so it means the resolution is (32x32x3) vs MNIST (28x28x1)

### Load library and plot sample data

```python
from keras.datasets import cifar10

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalized data to range (0, 1):
X_train, X_test = X_train/255, X_test/255


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
for i in range(49):
    plt.subplot(7,7,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i])
    # The CIFAR labels happen to be arrays, which is why you need the extra index    
    plt.xlabel(class_names[y_train[i][0]])
plt.show()
```
![image](https://user-images.githubusercontent.com/43855029/192858299-524f2e86-dd50-4bda-b227-587bde8ddd4d.png)


Next we do the preprocessing and construct/train the image classification model

```python
# Flatten the input data
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2]*X_train.shape[3])
X_test  = X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2]*X_test.shape[3])

# Normalize input data
X_train = X_train/X_train.max()
X_test = X_test/X_test.max()

# Categorical Encoding
import tensorflow.keras as keras
num_categories = 10 # Ranges from 0-9

y_train = keras.utils.to_categorical(y_train,num_categories)
y_test = keras.utils.to_categorical(y_test,num_categories)

# Create Keras Sequential model with 2 Dense Layer (first layer with 32 hidden units and second layer with 16 hidden units, same as MNIST problem). Pay attention to **input_shape**

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(units=32,activation='relu',input_shape=(3072,)))
model.add(Dense(units=16,activation='relu'))
model.add(Dense(units = num_categories,activation='softmax'))

# Compile and Train model
model.compile(optimizer="sgd",loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(X_train,y_train,epochs=10,verbose=1,validation_data=(X_test,y_test))
```

![image](https://user-images.githubusercontent.com/43855029/192870552-ad4e4687-ac96-4f09-9dea-a5b769af70a3.png)


```python
from matplotlib import pyplot as plt

def plot_acc_loss(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='best')

    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')

    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='best')

    plt.show()
    
plot_acc_loss(history)
```

![image](https://user-images.githubusercontent.com/43855029/192873763-ddc67e7a-97bb-4ff9-85c5-d61f98b400e0.png)


Evaluate on testing data

```python
import numpy as np
predictions = model.predict(X_test)
ypreds = np.argmax(predictions, axis=1)

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_test[i].reshape(32,32,3))
    plt.title(class_names[ypreds[i]])
plt.show()
```

![image](https://user-images.githubusercontent.com/43855029/192872164-6cd34a8d-d094-49cc-9c19-45a338129c8e.png)

### Discussions:
- The regular Dense or Vanila Neural Network does not work well on real image data as in CIFAR10.
- The best accuracy is about 0.44 for both training and testing set which is much lower than MNIST data
- How can we improve the model result?

![image](https://user-images.githubusercontent.com/43855029/192873251-b4644c24-81bf-4d14-8bf7-8650cf75dead.png)

- Next Let's move on to the next chapter, where we will learn more about Convolutional Neural Network or CNN

