---
title: "Convolution Neural Network for image classification: CIFAR10"
teaching: 20
exercises: 0
questions:
- "How to train a CNN model with Keras"
objectives:
- "Master Keras"
keypoints:
- "CNN, keras, CIFAR10"
---
# Convolutional Neural Network - CNN

- CNNs are one type of ANN which utilize the neuron, kernel, activation function.
- Inputs must be in images (or assumed to be images)
- Using Forward & Backpropagation technique with certain property to process it faster
- CNNs best for object detection, image classification, computer vision

![image](https://user-images.githubusercontent.com/43855029/165555812-c94fa369-193f-42e7-ba4b-b4e9a5fd7f67.png)

## Architecture of CNNs

![image](https://user-images.githubusercontent.com/43855029/129789560-452539b8-06c7-4a3b-8543-c2f6e5a6f9c6.png)
[Source](http://henrysprojects.net/projects/conv-net.html)

- A basic CNNs typically consists of Convolution Layers, Max Pooling Layers and fully connected Layer (Dense) before output layer
- A simple image can be simply flatten into 1D vector and driven through the regular fully connected NN. However, this requires lot of computational power if the image is large and has more color.
- Therefore, Convolution Layers and  Max Pooling help to reduce the size of the images but preserve the quality/structure of input images

### Convolutional Layer (CNN or ConvNet)

#### Hyperparameter: Depth (L)

- Take a look at the simple gray scale image below which contains 10 pixels on width & height. The color scale has only 2 values (black & white) or (binary -1 and 1), there fore the size of the following image is 10x10x1 (L=1):

![image](https://user-images.githubusercontent.com/43855029/129790068-408bbad8-8752-4153-9ce3-9099cae1995a.png)

- However, regular image contains colors [RGB](https://www.rapidtables.com/web/color/RGB_Color.html) with each color scale ranges from 0-255, making the size of each image is: n x n x 3 (n = number of pixel). (L=3)

![image](https://user-images.githubusercontent.com/43855029/129623983-173558ba-45f5-4a42-972d-a6252f7695e0.png)

#### Hyperparameter: Kernel (K) and Filter 

- The dot product between 2 matrices can be represented:

![image](https://user-images.githubusercontent.com/43855029/165556669-78f8f758-4ec4-4eb3-862f-780f5dfd8bcb.png)

- There are different Filters that can be applied:
  + Blur filter:
  
    ![image](https://user-images.githubusercontent.com/43855029/165556799-86bf38c1-897e-4ba5-8824-4a8b7e718c0c.png)
    
  + Sharp filter:
   
   ![image](https://user-images.githubusercontent.com/43855029/165556884-f1ac8ac9-3145-435f-8e38-09a0799cf695.png)

  + Edge dectetion filter:
    
    ![image](https://user-images.githubusercontent.com/43855029/165556949-783dff9e-45a9-42a4-a1f2-9a8997a03c12.png)

- CNN uses the Convolved Feature to reduce the image size by dot product with given kernel K.

- The image reduction without losing features and easier to process for good prediction

![image](https://user-images.githubusercontent.com/43855029/129624312-db0f2ce1-4767-4a18-9a02-f5cee4c6cfe5.png)

- So for 3 channel RGB colors, the image size have been reduced:

![image](https://user-images.githubusercontent.com/43855029/129624564-96d6d7e4-6409-4775-ad9d-2bf133fa0396.png)

- In other word, the convoluted image from RGB image would look like:

![image](https://user-images.githubusercontent.com/43855029/129791297-fae899e5-1745-4fa0-b348-1785dea769ea.png)

#### Hyperparameter: Stride (S):

Stride tuned for the compression of images and video data

  ![image](https://user-images.githubusercontent.com/43855029/165557343-70ee33bb-5820-4ca5-bb05-18ef6d7acaa9.png)

#### Hyperparameter: Padding (P):

- The pixels located on the corners and the edges are used much less than those in the middle => the information on borders and edges are note preserved
- Padding which add 0 around images to avoid lossing edge information

![image](https://user-images.githubusercontent.com/43855029/165557541-b4eaf3ed-d2f3-4eb1-aa02-a1857bf184e8.png)


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

### Batch Normalization Layer

- A process to make Deep neural networks faster and more stable through adding extra layers in a deep neural network.
- The new layer performs the standardizing and normalizing operations on the input of a layer coming from a previous layer.
- Normalize the amounts of weights trained between layers during training
- It usually goes after Conv2D layers or after Dense layer 

for example:

```python
model.add(Conv2D(75, (3, 3), strides=1, activation="relu", input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2))
```

### Dropout Layer

- Dropout is a regularization method that approximates training a large number of neural networks with different architectures in parallel.
- Dropout helps to avoid Overfitting
- Dropout is implemented per layer in the NN
- Dropout is not used after training when making a prediction with the fit network.

for example, randomly shutoff 20% neuron:

```python
model.add(Conv2D(75, (3, 3), strides=1, activation="relu", input_shape=(28, 28, 1)))
model.add(Dropout(0.2))
```

More information can be found [here](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)

## A sample of CNN model

![image](https://user-images.githubusercontent.com/43855029/165558822-d1024101-c76c-43bb-bf95-d381ba55bff9.png)

### Letnet-5 (1998) by Yann LeCun

- LeNet-5 is designed for handwritten and machine-printed character recognition
- Input image size is 32x32 pixels and having 1 channel color
- Total parameters: 60k
- Activation function: tanh

![image](https://user-images.githubusercontent.com/43855029/165558955-fc1e5a29-961a-41ee-bbb7-c4dd859350b5.png)

- It can be represented as:

```python
model = Sequential()
model.add(Conv2D(6, (5, 5), strides=(1, 1), activation=‘tanh’, padding=“valid”, input_shape=(32, 32, 1)))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(16, (5, 5), strides=(1, 1), activation=‘tanh’, padding=“valid”))
model.add(AveragePooling2D(pool_size=(2,2),strides=(2,2)
model.add(Conv2D(120, (5, 5), strides=(1, 1), activation=‘tanh’, padding=“valid”))
model.add(Flatten())
model.add(Dense(84,activation=‘tanh’))
model.add(Dropout(0.2))
model.add(Dense(10,activation=‘softmax’))
```

### Alex-Net (2012) by Hinton and Alex Krizhevsky

- AlexNet won the 2012 ImageNet challenge 
- Input images size is 227x227 pixels in 3 channel color RGB
- Total parameters: 60 millions
- Activation: ReLU

![image](https://user-images.githubusercontent.com/43855029/165559539-8bd75470-36c1-45d2-a4d0-e593b74e2e5c.png)

- Sample output of AlexNet:

![image](https://user-images.githubusercontent.com/43855029/165559605-49dcf70a-6c8d-418f-ba8d-2a60fba9d32f.png)

### VGG16 (2014)

- VGG16 runner up of 2014 ImageNet challenge 
- Image size is 224x224x3
- 16 layers: 13 ConvNet, 3 Fully Connected
- Total Parameters: 130M

![image](https://user-images.githubusercontent.com/43855029/165559808-2e7005a0-bfd1-48d7-9f58-5e779a5d9491.png)

### GoogleNet (2014)

- GoogleNet won the 2014 ImageNet challenge 
- Introduced Inception Network
- 22 layers deep with 27 pooling layers  and 9 inception models

![image](https://user-images.githubusercontent.com/43855029/165559901-fba35b54-a09a-46e8-8967-47e2e0d9f8c7.png)

# Application of CNN in image classification

## The CIFAR10 database

In this chapter, we are using CIFAR10 database with additional layer of Conv2D.

### Importing libraries
```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
```

### Import convolution, max pooling and flatten as mentioned above:
```python
from tensorflow.keras.layers import Conv2D # convolutional layers to reduce image size
from tensorflow.keras.layers import MaxPooling2D # Max pooling layers to further reduce image size
from tensorflow.keras.layers import AveragePooling2D # Max pooling layers to further reduce image size
from tensorflow.keras.layers import Flatten # flatten data from 2D to column for Dense layer
```

### Load CIFAR10

```python
from tensorflow.keras.datasets import cifar10
# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# Normalized data to range (0, 1):
X_train, X_test = X_train/X_train.max(), X_test/X_test.max()
```

### Using One Hot Encoding from Keras to convert the label:

```python
num_categories = 10 # Ranges from 0-9
input_shape = (32,32,3) # 32 pixels with 3D color scale

y_train = tf.keras.utils.to_categorical(y_train,num_categories)
y_test = tf.keras.utils.to_categorical(y_test,num_categories)
```

### Construct Convolutional Neural Network
- For Convolution front end, starting with kernel size (3,3) with a number of filter 10 followed by Max Pooling Layer with pool_size = (2,2).
- The 2D data after two Max Pooling layer is flatten directly.

```python
model = Sequential()
model.add(Conv2D(8, (3, 3), strides=(1, 1), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(100, activation='relu'))
#Output layer contains 10 different number from 0-9
model.add(Dense(10, activation='softmax'))
```

### Compile model

```python
model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])                            
```

### Train model

Fit the model

```python
# fit the model
history = model.fit(X_train, y_train, epochs=10, 
                    validation_data=(X_test, y_test))
```

```
Epoch 1/10
   1/1563 [..............................] - ETA: 0s - loss: 2.3374 - accuracy: 0.1875WARNING:tensorflow:From /users/tuev/.conda/envs/ML_SKLN/lib/python3.6/site-packages/tensorflow/python/ops/summary_ops_v2.py:1277: stop (from tensorflow.python.eager.profiler) is deprecated and will be removed after 2020-07-01.
Instructions for updating:
use `tf.profiler.experimental.stop` instead.
   2/1563 [..............................] - ETA: 1:35 - loss: 2.3676 - accuracy: 0.1406WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0051s vs `on_train_batch_end` time: 0.1163s). Check your callbacks.
1563/1563 [==============================] - 5s 3ms/step - loss: 1.5274 - accuracy: 0.4561 - val_loss: 1.3043 - val_accuracy: 0.5437
Epoch 2/10
1563/1563 [==============================] - 5s 3ms/step - loss: 1.2409 - accuracy: 0.5636 - val_loss: 1.2332 - val_accuracy: 0.5674
Epoch 3/10
1563/1563 [==============================] - 5s 3ms/step - loss: 1.1386 - accuracy: 0.5986 - val_loss: 1.1798 - val_accuracy: 0.5854
Epoch 4/10
1563/1563 [==============================] - 5s 3ms/step - loss: 1.0740 - accuracy: 0.6230 - val_loss: 1.1629 - val_accuracy: 0.5898
Epoch 5/10
1563/1563 [==============================] - 5s 3ms/step - loss: 1.0168 - accuracy: 0.6413 - val_loss: 1.1529 - val_accuracy: 0.5975
Epoch 6/10
1563/1563 [==============================] - 5s 3ms/step - loss: 0.9661 - accuracy: 0.6588 - val_loss: 1.1583 - val_accuracy: 0.5995
Epoch 7/10
1563/1563 [==============================] - 5s 3ms/step - loss: 0.9240 - accuracy: 0.6755 - val_loss: 1.1451 - val_accuracy: 0.6039
Epoch 8/10
1563/1563 [==============================] - 5s 3ms/step - loss: 0.8831 - accuracy: 0.6894 - val_loss: 1.1701 - val_accuracy: 0.6020
Epoch 9/10
1563/1563 [==============================] - 5s 3ms/step - loss: 0.8506 - accuracy: 0.7017 - val_loss: 1.1574 - val_accuracy: 0.6069
Epoch 10/10
1563/1563 [==============================] - 5s 3ms/step - loss: 0.8126 - accuracy: 0.7129 - val_loss: 1.1547 - val_accuracy: 0.6078
```

### Evaluate the output

Visualize the training/testing accuracy:

```python
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

plot_acc_loss(model_CNN)
```

![image](https://user-images.githubusercontent.com/43855029/192877213-0bb285a2-87e7-4e3c-90d1-b559706496dc.png)

### Discussions:
- We can see the accuracy increased from 0.44 to 0.7 for training and 0.6 for testing

## Application of Letnet-5 (1998) to Fashion MNIST?

- AlexNet is the name of CNN architecture, designed by Alex Krizhevsky in collaboration with Ilya Sutskever and Geoffrey Hinton.
- AlexNet competed in the ImageNet Challenge 2012, the network achieved a top-5 error of 15.3%, more than 10.8 percentage points lower than that of the runner up. 
- The original paper's primary result was that the depth of the model was essential for its high performance, which was computationally expensive, but made feasible due to the utilization of graphics processing units (GPUs) during training

![image](https://user-images.githubusercontent.com/43855029/162292152-b2fab741-1ddf-477a-9f07-140c4ea2421d.png)

### The architecture of AlexNet

```python
model = Sequential()
# C1 Convolution Layer
model.add(Conv2D(6, (3, 3), strides=(1, 1), activation='tanh', padding='same', input_shape=(28, 28, 1)))
# S2 Pooling Layer
model.add(AveragePooling2D(pool_size=(2,2),strides=(2,2)))
# C3 Convolution Layer
model.add(Conv2D(16, (5, 5), strides=(1, 1), activation='tanh', padding='valid'))
# S4 Pooling Layer
model.add(AveragePooling2D(pool_size=(2,2),strides=(2,2)))
# C5 Convolution Layer
model.add(Conv2D(120, (5, 5), strides=(1, 1), activation='tanh', padding='valid'))
# Flatten to MLP          
model.add(Flatten())
# F6 Fully connected Layer         
model.add(Dense(84,activation='tanh'))
# Output layer          
model.add(Dense(10,activation='softmax'))
# Compile model
model.compile(optimizer='sgd', loss='categorical_crossentropy',  metrics=['accuracy'])       
```

### Fit the model

```python
history = model.fit(X_train, y_train, epochs=10, batch_size=128,
                    validation_data=(X_test, y_test))       
```                    

```
Epoch 1/10
469/469 [==============================] - 3s 7ms/step - loss: 1.3290 - accuracy: 0.5772 - val_loss: 0.9032 - val_accuracy: 0.6799
Epoch 2/10
469/469 [==============================] - 3s 6ms/step - loss: 0.7835 - accuracy: 0.7273 - val_loss: 0.7265 - val_accuracy: 0.7414
Epoch 3/10
469/469 [==============================] - 3s 6ms/step - loss: 0.6682 - accuracy: 0.7593 - val_loss: 0.6591 - val_accuracy: 0.7625
Epoch 4/10
469/469 [==============================] - 3s 6ms/step - loss: 0.6145 - accuracy: 0.7792 - val_loss: 0.6193 - val_accuracy: 0.7760
Epoch 5/10
469/469 [==============================] - 3s 6ms/step - loss: 0.5791 - accuracy: 0.7919 - val_loss: 0.5874 - val_accuracy: 0.7876
Epoch 6/10
469/469 [==============================] - 3s 6ms/step - loss: 0.5520 - accuracy: 0.8022 - val_loss: 0.5656 - val_accuracy: 0.7943
Epoch 7/10
469/469 [==============================] - 3s 6ms/step - loss: 0.5304 - accuracy: 0.8103 - val_loss: 0.5471 - val_accuracy: 0.8018
Epoch 8/10
469/469 [==============================] - 3s 6ms/step - loss: 0.5125 - accuracy: 0.8168 - val_loss: 0.5314 - val_accuracy: 0.8062
Epoch 9/10
469/469 [==============================] - 3s 6ms/step - loss: 0.4970 - accuracy: 0.8229 - val_loss: 0.5162 - val_accuracy: 0.8123
Epoch 10/10
469/469 [==============================] - 3s 6ms/step - loss: 0.4836 - accuracy: 0.8277 - val_loss: 0.5033 - val_accuracy: 0.8174
```

### Visulize the output

```python
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
![image](https://user-images.githubusercontent.com/43855029/162294211-75b48704-faf8-4389-85d7-6fda052c08e9.png)


Which one is better? ALexNet using tanh or the previous using ReLU activation function?

### Visulize the output with images and its labels using AlexNet:

```python
predictions = model.predict(X_test)
ypreds = np.argmax(predictions, axis=1)

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_test[i])
    plt.title(class_names[ypreds[i]])
plt.show()
```

![image](https://user-images.githubusercontent.com/43855029/162297902-6709919f-78a6-4787-846b-6bfa2399e5e0.png)
