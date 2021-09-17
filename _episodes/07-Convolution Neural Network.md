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


## Application of CNN in hand writing recognition.

### The MNIST database of handwritten digits
- The [MNIST](http://yann.lecun.com/exdb/mnist/) database of handwritten digits has training/test set of 60,000/10,000 samples.
- The digits have been normalized and centered in a fixed-size image
- It is good database for pattern recognition and image classification task (the entire data is clean and ready for use).
- Each image were centered in 28 x 28 pixel with RGB color range from 0-255
- Sample MNIST data:

![image](https://user-images.githubusercontent.com/43855029/133843977-38998202-5688-4053-9acf-93ac7b48da21.png)
[Source](https://commons.wikimedia.org/wiki/File:MnistExamples.png)

### Importing libraries
```python
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
```

### Import convolution, max pooling and flatten as mentioned above:
```python
from keras.layers.convolutional import Conv2D # convolutional layers to reduce image size
from keras.layers.convolutional import MaxPooling2D # Max pooling layers to further reduce image size
from keras.layers import Flatten # flatten data from 2D to column for Dense layer
```

### Load MINST data
```python
from keras.datasets import mnist

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

Sample ploting:
```python
import matplotlib.pyplot as plt
# pick a sample to plot
sample = 2
image = X_train[sample]
# plot the sample
fig = plt.figure
plt.imshow(image)
plt.show()
```
![image](https://user-images.githubusercontent.com/43855029/133841615-5e719ba3-19a7-4299-bfc7-3bf166032d98.png)


```python
# Reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
```

#### Normalized MINST data
```python
X_train = X_train / 255 
X_test = X_test / 255
#255 is the max range of RGB color
```

### Convert output data to binary
```python
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

no_class = y_test.shape[1] # number of categories from 0-9
```

### Construct Convolutional Neural Network
- For Convolution front end, starting with kernel size (5,5) with a number of filter 10 followed by Max Pooling Layer with pool_size = (2,2).
- The 2D data after first Max Pooling layer is flatten directly.

```python
model = Sequential()
model.add(Conv2D(10, (5, 5), strides=(1, 1), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(no_class, activation='softmax'))

# compile model
model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])
```

### Train model

Fit the model

```python
# fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
```

```
Epoch 1/10
300/300 - 4s - loss: 0.3221 - accuracy: 0.9114 - val_loss: 0.1155 - val_accuracy: 0.9679
Epoch 2/10
300/300 - 3s - loss: 0.0970 - accuracy: 0.9721 - val_loss: 0.0777 - val_accuracy: 0.9757
Epoch 3/10
300/300 - 3s - loss: 0.0655 - accuracy: 0.9809 - val_loss: 0.0581 - val_accuracy: 0.9817
Epoch 4/10
300/300 - 3s - loss: 0.0515 - accuracy: 0.9844 - val_loss: 0.0522 - val_accuracy: 0.9826
Epoch 5/10
300/300 - 3s - loss: 0.0422 - accuracy: 0.9876 - val_loss: 0.0486 - val_accuracy: 0.9846
Epoch 6/10
300/300 - 3s - loss: 0.0362 - accuracy: 0.9890 - val_loss: 0.0467 - val_accuracy: 0.9842
Epoch 7/10
300/300 - 3s - loss: 0.0310 - accuracy: 0.9905 - val_loss: 0.0419 - val_accuracy: 0.9869
Epoch 8/10
300/300 - 3s - loss: 0.0266 - accuracy: 0.9918 - val_loss: 0.0437 - val_accuracy: 0.9857
Epoch 9/10
300/300 - 3s - loss: 0.0230 - accuracy: 0.9930 - val_loss: 0.0425 - val_accuracy: 0.9851
Epoch 10/10
300/300 - 3s - loss: 0.0209 - accuracy: 0.9937 - val_loss: 0.0413 - val_accuracy: 0.9855
<tensorflow.python.keras.callbacks.History at 0x146d080ab2e0>
```

### Evaluate the output

Visualize the training/testing accuracy:

```python
fig = plt.figure(figsize=(8, 10), dpi=80)
plt.subplot(2,1,1)
plt.plot(model_CNN.history['accuracy'],"b-o")
plt.plot(model_CNN.history['val_accuracy'],"r-d")
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.legend(['train', 'test'])

plt.subplot(2,1,2)
plt.plot(model_CNN.history['loss'],"b-o")
plt.plot(model_CNN.history['val_loss'],"r-d")
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.tight_layout()
fig
```

![image](https://user-images.githubusercontent.com/43855029/133845639-172800ef-ec18-4bf8-a779-1367a3b85519.png)

### Save & reload CNN model
Save model:

```python
model.save('/home/tuev/CNN_MNIST.keras')
```

Reload model:
```python
model = keras.models.load_model('/home/tuev/CNN_MNIST.keras')
```

### Evaluate model with testing data
```python
test_loss, test_accuracy = model1.evaluate(X_test, y_test, batch_size=64)
print('Test loss: %.4f accuracy: %.4f' % (test_loss, test_accuracy))
```

```
157/157 [==============================] - 0s 2ms/step - loss: 0.0428 - accuracy: 0.9884
Test loss: 0.0428 accuracy: 0.9884
```

The accuracy rate is 0.9884 for testing data means there are 9884 right classification based on 10,000 sample of testing data
