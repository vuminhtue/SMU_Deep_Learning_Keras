---
title: "Convolution Neural Network for image classification: Fashion MNIST"
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

### Convolutional Layer (CNN or ConvNet)

- Take a look at the simple gray scale image below which contains 10 pixels on width & height. The color scale has only 2 values (black & white) or (binary -1 and 1), there fore the size of the following image is 10x10x1:

![image](https://user-images.githubusercontent.com/43855029/129790068-408bbad8-8752-4153-9ce3-9099cae1995a.png)

- However, regular image contains colors [RGB](https://www.rapidtables.com/web/color/RGB_Color.html) with each color scale ranges from 0-255, making the size of each image is: n x n x 3 (n = number of pixel).

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


# Application of CNN in image classification

## The Fashion MNIST database
The original MNIST dataset contains a lot of handwritten digits. Members of the AI/ML/Data Science community love this dataset and use it as a benchmark to validate their algorithms. In fact, MNIST is often the first dataset researchers try. "If it doesn't work on MNIST, it won't work at all", they said. "Well, if it does work on MNIST, it may still fail on others."

- Fashion MNIST is very similar to digit MNIST
- It consisting of a training set of 60,000 examples and a test set of 10,000 examples.
- Each example is a 28x28 grayscale image, associated with a label from 10 classes. 
- Zalando intends Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. 
- It shares the same image size and structure of training and testing splits.

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
from tensorflow.keras.layers import Conv2D # convolutional layers to reduce image size
from tensorflow.keras.layers import MaxPooling2D # Max pooling layers to further reduce image size
from tensorflow.keras.layers import AveragePooling2D # Max pooling layers to further reduce image size
from tensorflow.keras.layers import Flatten # flatten data from 2D to column for Dense layer
```

### Load Fashion MNIST

```python
from tensorflow.keras.datasets import fashion_mnist
# load data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
# Normalized data to range (0, 1):
X_train, X_test = X_train/X_train.max(), X_test/X_test.max()

X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)
```

Sample ploting:
```python
# Sample ploting:

class_names = ['t-shirt/top', 'trouser', 'pullover', 'dress', 'coat',
               'sandall', 'shirt', 'sneaker', 'bag', 'ankle boot']

import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
for i in range(49):
    plt.subplot(7,7,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i])
    # The FMNIST labels happen to be arrays, which is why you need the extra index    
    plt.title(class_names[y_train[i]])
plt.show()

```

![image](https://user-images.githubusercontent.com/43855029/162291086-fe8203bf-ebf2-485d-8e08-eb39a0a21103.png)


Using One Hot Encoding from Keras to convert the label:

```python
num_categories = 10 # Ranges from 0-9
input_shape = (28,28,1) # 28 pixels with 1D color scale

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
model.add(Dense(28, activation='relu'))
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
1875/1875 [==============================] - 10s 5ms/step - loss: 0.7601 - accuracy: 0.7301 - val_loss: 0.5876 - val_accuracy: 0.7793
Epoch 2/10
1875/1875 [==============================] - 9s 5ms/step - loss: 0.5157 - accuracy: 0.8159 - val_loss: 0.5114 - val_accuracy: 0.8169
Epoch 3/10
1875/1875 [==============================] - 9s 5ms/step - loss: 0.4652 - accuracy: 0.8346 - val_loss: 0.4645 - val_accuracy: 0.8399
Epoch 4/10
1875/1875 [==============================] - 9s 5ms/step - loss: 0.4337 - accuracy: 0.8476 - val_loss: 0.4469 - val_accuracy: 0.8437
Epoch 5/10
1875/1875 [==============================] - 9s 5ms/step - loss: 0.4080 - accuracy: 0.8558 - val_loss: 0.4232 - val_accuracy: 0.8517
Epoch 6/10
1875/1875 [==============================] - 9s 5ms/step - loss: 0.3870 - accuracy: 0.8637 - val_loss: 0.4086 - val_accuracy: 0.8554
Epoch 7/10
1875/1875 [==============================] - 9s 5ms/step - loss: 0.3679 - accuracy: 0.8703 - val_loss: 0.3773 - val_accuracy: 0.8685
Epoch 8/10
1875/1875 [==============================] - 9s 5ms/step - loss: 0.3529 - accuracy: 0.8745 - val_loss: 0.3680 - val_accuracy: 0.8710
Epoch 9/10
1875/1875 [==============================] - 9s 5ms/step - loss: 0.3395 - accuracy: 0.8798 - val_loss: 0.3526 - val_accuracy: 0.8751
Epoch 10/10
1875/1875 [==============================] - 9s 5ms/step - loss: 0.3278 - accuracy: 0.8839 - val_loss: 0.3484 - val_accuracy: 0.8771
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

plot_acc_loss(history)
```

![image](https://user-images.githubusercontent.com/43855029/162291733-4a32caf2-3209-492a-b2b2-601a42a3deb8.png)


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

### Visulize the output with images and its labels:

```python
predictions = model_new.predict(X_test)
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
