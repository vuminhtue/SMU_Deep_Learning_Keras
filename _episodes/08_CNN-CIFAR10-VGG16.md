---
title: "Convolution Neural Network for image classification: CIFAR10"
teaching: 20
exercises: 0
questions:
- "How to train a CNN model with Keras"
objectives:
- "Master Keras"
keypoints:
- "CNN, keras"
---

# Application of CNN in image classification

### The CIFAR10 database
- The [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) database consisting 60,000 color images with 10 different classes
- Each image has 32 x 32 pixels with color range from 0-255
- It is good database for pattern recognition and image classification task (the entire data is clean and ready for use).
- The dataset was divided into 50,000 images for training and 10,000 images for testing
- The 10 [classes](https://keras.io/api/datasets/cifar10/) are **airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck**
- Sample CIFAR10 data:

![image](https://user-images.githubusercontent.com/43855029/134049153-99879363-c761-4b1d-b378-78186024bb95.png)[Source](https://www.cs.toronto.edu/~kriz/cifar.html)

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

### Load CIFAR10 data
```python
from keras.datasets import cifar10

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalized data to range (0, 1):
X_train, X_test = X_train/255, X_test/255
X_train.shape
X_test.shape
y_train.shape
y_test.shape
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


Using One Hot Encoding from Keras to convert the label:

```python
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_train.shape
y_test.shape
```

### Construct Convolutional Neural Network
- For Convolution front end, starting with kernel size (3,3) with a number of filter 10 followed by Max Pooling Layer with pool_size = (2,2).
- The 2D data after two Max Pooling layer is flatten directly.

```python
model = Sequential()
model.add(Conv2D(8, (3, 3), strides=(1, 1), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(100, activation='relu'))
#Output layer contains 10 different number from 0-9
model.add(Dense(10, activation='softmax'))

# compile model
model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])                            
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
1563/1563 [==============================] - 7s 4ms/step - loss: 1.5531 - accuracy: 0.4444 - val_loss: 1.3197 - val_accuracy: 0.5285
Epoch 2/10
1563/1563 [==============================] - 6s 4ms/step - loss: 1.2657 - accuracy: 0.5534 - val_loss: 1.2416 - val_accuracy: 0.5575
Epoch 3/10
1563/1563 [==============================] - 6s 4ms/step - loss: 1.1759 - accuracy: 0.5868 - val_loss: 1.2053 - val_accuracy: 0.5776
Epoch 4/10
1563/1563 [==============================] - 6s 4ms/step - loss: 1.1125 - accuracy: 0.6091 - val_loss: 1.2001 - val_accuracy: 0.5796
Epoch 5/10
1563/1563 [==============================] - 6s 4ms/step - loss: 1.0671 - accuracy: 0.6244 - val_loss: 1.1442 - val_accuracy: 0.5966
Epoch 6/10
1563/1563 [==============================] - 6s 4ms/step - loss: 1.0343 - accuracy: 0.6374 - val_loss: 1.1755 - val_accuracy: 0.5926
Epoch 7/10
1563/1563 [==============================] - 6s 4ms/step - loss: 1.0012 - accuracy: 0.6477 - val_loss: 1.1348 - val_accuracy: 0.6029
Epoch 8/10
1563/1563 [==============================] - 6s 4ms/step - loss: 0.9710 - accuracy: 0.6615 - val_loss: 1.1379 - val_accuracy: 0.6060
Epoch 9/10
1563/1563 [==============================] - 6s 4ms/step - loss: 0.9397 - accuracy: 0.6699 - val_loss: 1.1469 - val_accuracy: 0.5988
Epoch 10/10
1563/1563 [==============================] - 6s 4ms/step - loss: 0.9208 - accuracy: 0.6772 - val_loss: 1.1538 - val_accuracy: 0.6080
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

![image](https://user-images.githubusercontent.com/43855029/135329289-3e48cf23-ab0c-4a35-8074-b0a344e8dbbe.png)


### Save & reload CNN model
Save model:

```python
model.save('CNN_CIFAR10.keras')
```

Reload model:
```python
model_new = keras.models.load_model('CNN_CIFAR10.keras')
```

### Evaluate model with testing data
```python
test_loss, test_accuracy = model_new.evaluate(X_test, y_test, batch_size=64)
print('Test loss: %.4f accuracy: %.4f' % (test_loss, test_accuracy))
```

```
157/157 [==============================] - 0s 2ms/step - loss: 1.1538 - accuracy: 0.6080
Test loss: 1.1538 accuracy: 0.6080
```

The accuracy rate is 0.6080 for testing data means there are 6080 right classification based on 10,000 sample of testing data

### Visualize the output with the first 25 testing images

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
    plt.xlabel(class_names[ypreds[i]])
plt.show()
```

![image](https://user-images.githubusercontent.com/43855029/135329702-29eeb261-e7bb-4d93-bd6c-894e8360df16.png)

### Improving the performance?
Use more convolution and max pooling payer:

```python
model = Sequential()
model.add(Conv2D(8, (3, 3), strides=(1, 1), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Flatten())
model.add(Dense(100, activation='relu'))
#Output layer contains 10 different number from 0-9
model.add(Dense(10, activation='softmax'))

# compile model
model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])                            
```

# Using pre-trained model VGG16

- A widely used CNN architecture for ImageNet challange (top 5) 2014, developed at Visual Geometry Group (VGG) using 16 layers.
- It surpassed AlexNet by replacing large kernel-sized filters with multipled 3x3 kernel-sized filters one after another
- Until now, It is still one of the best vision architecture to date.

![image](https://user-images.githubusercontent.com/43855029/162279133-27b22ade-7aa7-4d3d-a4ef-1c8e36648015.png)

The architecture of VGG16:

![image](https://user-images.githubusercontent.com/43855029/162279702-f7694360-8db0-4df0-8932-2a43a6e7bce8.png)

In this workshop, we gonna use VGG16 to recognize some images downloaded from internet.

## Load VGG16

```python
from tensorflow.keras.applications import VGG16
 
# load the VGG16 network *pre-trained* on the ImageNet dataset
model = VGG16(weights="imagenet")
model.summary()
```

## Download some images:

Function to show images:

```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_image(image_path):
    image = mpimg.imread(image_path)
    print(image.shape)
    plt.imshow(image)
```

Download an image:

```python
!wget https://cdn.britannica.com/29/150929-050-547070A1/lion-Kenya-Masai-Mara-National-Reserve.jpg
```

Show image:

```python
show_image("lion-Kenya-Masai-Mara-National-Reserve.jpg")
```

![image](https://user-images.githubusercontent.com/43855029/162304581-dfc34da1-74e8-4bb7-a95b-9750c22a1da5.png)


## Processing input images to VGG16 input

```python
from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.applications.vgg16 import preprocess_input

def load_and_process_image(image_path):
    # Print image's original shape, for reference
    print('Original image shape: ', mpimg.imread(image_path).shape)
    
    # Load in the image with a target size of 224, 224
    image = image_utils.load_img(image_path, target_size=(224, 224))
    # Convert the image from a PIL format to a numpy array
    image = image_utils.img_to_array(image)
    # Add a dimension for number of images, in our case 1
    image = image.reshape(1,224,224,3)
    # Preprocess image to align with original ImageNet dataset
    image = preprocess_input(image)
    # Print image's shape after processing
    print('Processed image shape: ', image.shape)
    return image
```

## Prediction using VGG16

Now that we have our image in the right format, we can pass it into our model and get a prediction. We are expecting an output of an array of 1000 elements, which is going to be difficult to read. Fortunately, models loaded directly with Keras have yet another helpful method that will translate that prediction array into a more readable form.

Fill in the following function to implement the prediction:

```python
from tensorflow.keras.applications.vgg16 import decode_predictions

def readable_prediction(image_path):
    # Show image
    show_image(image_path)
    # Load and pre-process image
    image = load_and_process_image(image_path)
    # Make predictions
    predictions = model.predict(image)
    # Print predictions in readable form
    print('Predicted:', decode_predictions(predictions, top=3))
```

Predict the image:

```python
readable_prediction("lion-Kenya-Masai-Mara-National-Reserve.jpg")
```

![image](https://user-images.githubusercontent.com/43855029/162304749-49c9fae2-516e-405e-aadd-026e0eb01986.png)


