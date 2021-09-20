---
title: "Convolution Neural Network for image classification"
teaching: 20
exercises: 0
questions:
- "How to train a CNN model with Keras"
objectives:
- "Master Keras"
keypoints:
- "CNN, keras"
---
# Convolutional Neural Network - CNN


- CNNs are one type of ANN which utilize the neuron, kernel, activation function.
- Inputs must be in images (or assumed to be images)
- Using Forward & Backpropagation technique with certain property to process it faster
- CNNs best for object detection, image classification, computer vision

## Architecture of CNNs


![image](https://user-images.githubusercontent.com/43855029/129789560-452539b8-06c7-4a3b-8543-c2f6e5a6f9c6.png)
[Source](http://henrysprojects.net/projects/conv-net.html)

- A basic CNNs consists of Convolution Layers, Max Pooling Layers and fully connected Layer (Dense) before output layer
- A simple image can be simply flatten into 1D vector and driven through the regular fully connected NN. However, this requires lot of computational power if the image is large and has more color.
- Therefore, Convolution Layers and  Max Pooling

### Convolutional Neural Network (CNN or ConvNet)

- Take a look at the simple gray scale image below which contains 10 pixels on width & height. The color scale has only 2 values (black & white) or (binary -1 and 1), there fore the size of the following image is 10x10x1:

![image](https://user-images.githubusercontent.com/43855029/129790068-408bbad8-8752-4153-9ce3-9099cae1995a.png)

- However, regular image contains colors RGB with each color scale ranges from 0-255, making the size of each image is: n x n x 3 (n = number of pixel).

![image](https://user-images.githubusercontent.com/43855029/129623983-173558ba-45f5-4a42-972d-a6252f7695e0.png)

- CNN uses the Convolved Feature to reduce the image size by dot product with given kernel.
- The image reduction without losing features and easier to process for good prediction

![image](https://user-images.githubusercontent.com/43855029/129624312-db0f2ce1-4767-4a18-9a02-f5cee4c6cfe5.png)

- So for 3 channel RGB colors, the image size have been reduced:

![image](https://user-images.githubusercontent.com/43855029/129624564-96d6d7e4-6409-4775-ad9d-2bf133fa0396.png)

- In other word, the convoluted image from RGB image would look like:

![image](https://user-images.githubusercontent.com/43855029/129791297-fae899e5-1745-4fa0-b348-1785dea769ea.png)


### Pooling Layer
- Similar to the Convolutional Layer, the Pooling layer is responsible for reducing the spatial size of the Convolved Feature.
- This is to decrease the computational power required to process the data through dimensionality reduction
- Two types of Pooling: Max Pooling & Average Pooling.

![image](https://user-images.githubusercontent.com/43855029/129624678-75532145-0e90-48d5-9703-c8ee626aa7f4.png)

In which Max Pooling performs a lot better than Average Pooling.

- The image after Max Pooling layer would look like:

![image](https://user-images.githubusercontent.com/43855029/129791581-5d9fa47d-1390-44c2-b86a-f66273a9f7ca.png)


### Flatten Layer
- Once the images have passed through Convolution Layer and Pooling Layer, its size has been reduced greatly and ready for MLP training (or to another Convolution steps).
- The image is then flatten to a column vector and passed through feed-forward NN and BackPropagation applied to every iteration.
- Softmax activation function is applied to classified the multi-output

More information can be found [here](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)


## Application of CNN in image classification

### The CIFAR10 database
- The [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) database consisting 60,000 color images with 10 different classes
- Each image has 32 x 32 pixels with color range from 0-255
- It is good database for pattern recognition and image classification task (the entire data is clean and ready for use).
- The dataset was divided into 50,000 images for training and 10,000 images for testing
- The 10 classes are **airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck**
- Sample CIFAR10 data:

![image](https://user-images.githubusercontent.com/43855029/134049153-99879363-c761-4b1d-b378-78186024bb95.png)[Source](https://www.cs.toronto.edu/~kriz/cifar.html)

### Importing libraries
```python
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

```

### Import convolution, max pooling and flatten as mentioned above:
```python
from keras.layers.convolutional import Conv2D # convolutional layers to reduce image size
from keras.layers.convolutional import MaxPooling2D # Max pooling layers to further reduce image size
from keras.layers import Flatten # flatten data from 2D to column for Dense layer
```

### Load CIFAR10 data
```python
from keras.datasets import cifar10

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
```

Sample ploting:
```python
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

![image](https://user-images.githubusercontent.com/43855029/134049444-f95cd292-9b5f-40f9-852c-6bbe0a724d78.png)


### Construct Convolutional Neural Network
- For Convolution front end, starting with kernel size (3,3) with a number of filter 10 followed by Max Pooling Layer with pool_size = (2,2).
- The 2D data after two Max Pooling layer is flatten directly.

```python
model = Sequential()
model.add(Conv2D(10, (3, 3), strides=(1, 1), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Flatten())
model.add(Dense(100, activation='relu'))
#Output layer contains 10 different number from 0-9
model.add(Dense(10, activation='softmax'))

# compile model
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),              
              metrics=['accuracy'])
              
```

### Train model

Fit the model

```python
# fit the model
model_CNN = model.fit(X_train, y_train, epochs=10, 
                    validation_data=(X_test, y_test))
```

```
Epoch 1/10
1563/1563 [==============================] - 10s 7ms/step - loss: 2.0117 - accuracy: 0.3432 - val_loss: 1.5362 - val_accuracy: 0.4350
Epoch 2/10
1563/1563 [==============================] - 10s 6ms/step - loss: 1.4181 - accuracy: 0.4910 - val_loss: 1.3415 - val_accuracy: 0.5173
Epoch 3/10
1563/1563 [==============================] - 10s 6ms/step - loss: 1.2498 - accuracy: 0.5558 - val_loss: 1.2070 - val_accuracy: 0.5794
Epoch 4/10
1563/1563 [==============================] - 10s 6ms/step - loss: 1.1396 - accuracy: 0.6031 - val_loss: 1.1446 - val_accuracy: 0.6008
Epoch 5/10
1563/1563 [==============================] - 10s 6ms/step - loss: 1.0403 - accuracy: 0.6388 - val_loss: 1.0624 - val_accuracy: 0.6325
Epoch 6/10
1563/1563 [==============================] - 10s 6ms/step - loss: 0.9659 - accuracy: 0.6657 - val_loss: 1.0405 - val_accuracy: 0.6483
Epoch 7/10
1563/1563 [==============================] - 10s 6ms/step - loss: 0.8978 - accuracy: 0.6904 - val_loss: 1.0140 - val_accuracy: 0.6605
Epoch 8/10
1563/1563 [==============================] - 10s 6ms/step - loss: 0.8466 - accuracy: 0.7062 - val_loss: 1.0174 - val_accuracy: 0.6587
Epoch 9/10
1563/1563 [==============================] - 10s 6ms/step - loss: 0.7965 - accuracy: 0.7241 - val_loss: 1.0165 - val_accuracy: 0.6594
Epoch 10/10
1563/1563 [==============================] - 10s 6ms/step - loss: 0.7515 - accuracy: 0.7407 - val_loss: 1.0503 - val_accuracy: 0.6645
```

### Evaluate the output

Visualize the training/testing accuracy:

```python
fig = plt.figure(figsize=(8, 10), dpi=80)
plt.subplot(2,1,1)
plt.plot(history.history['accuracy'],"b-o")
plt.plot(history.history['val_accuracy'],"r-d")
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.legend(['train', 'test'])

plt.subplot(2,1,2)
plt.plot(history.history['loss'],"b-o")
plt.plot(history.history['val_loss'],"r-d")
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.tight_layout()
fig
```

![image](https://user-images.githubusercontent.com/43855029/134049936-c007a7b5-5dbf-4f23-b2f0-8c9ed7ab8de1.png)


### Save & reload CNN model
Save model:

```python
model.save('/home/tuev/CNN_CIFAR10.keras')
```

Reload model:
```python
model = keras.models.load_model('/home/tuev/CNN_CIFAR10.keras')
```

### Evaluate model with testing data
```python
test_loss, test_accuracy = model_CNN.evaluate(X_test, y_test, batch_size=64)
print('Test loss: %.4f accuracy: %.4f' % (test_loss, test_accuracy))
```

```
157/157 [==============================] - 0s 3ms/step - loss: 1.0503 - accuracy: 0.6645
Test loss: 1.0503 accuracy: 0.6645
```

The accuracy rate is 0.6645 for testing data means there are 6645 right classification based on 10,000 sample of testing data
