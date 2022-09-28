---
title: "MNIST with Dense Network"
teaching: 20
exercises: 0
questions:
- "Image Classification with MNIST data"
objectives:
- "MNIST, Dense Layer"
keypoints:
- "Classification training, MNIST, keras"
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
