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

model.summary()
```

```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 30, 30, 10)        280       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 15, 15, 10)       0         
 )                                                               
                                                                 
 flatten (Flatten)           (None, 2250)              0         
                                                                 
 dense (Dense)               (None, 100)               225100    
                                                                 
 dense_1 (Dense)             (None, 10)                1010      
                                                                 
=================================================================
Total params: 226,390
Trainable params: 226,390
Non-trainable params: 0
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
1563/1563 [==============================] - 6s 2ms/step - loss: 1.4979 - accuracy: 0.4661 - val_loss: 1.3041 - val_accuracy: 0.5401
Epoch 2/10
1563/1563 [==============================] - 3s 2ms/step - loss: 1.2271 - accuracy: 0.5668 - val_loss: 1.2155 - val_accuracy: 0.5687
Epoch 3/10
1563/1563 [==============================] - 3s 2ms/step - loss: 1.1153 - accuracy: 0.6101 - val_loss: 1.2015 - val_accuracy: 0.5777
Epoch 4/10
1563/1563 [==============================] - 3s 2ms/step - loss: 1.0302 - accuracy: 0.6395 - val_loss: 1.1256 - val_accuracy: 0.6108
Epoch 5/10
1563/1563 [==============================] - 3s 2ms/step - loss: 0.9635 - accuracy: 0.6622 - val_loss: 1.0992 - val_accuracy: 0.6172
Epoch 6/10
1563/1563 [==============================] - 3s 2ms/step - loss: 0.9080 - accuracy: 0.6826 - val_loss: 1.1595 - val_accuracy: 0.6001
Epoch 7/10
1563/1563 [==============================] - 3s 2ms/step - loss: 0.8541 - accuracy: 0.6997 - val_loss: 1.1124 - val_accuracy: 0.6259
Epoch 8/10
1563/1563 [==============================] - 3s 2ms/step - loss: 0.8037 - accuracy: 0.7167 - val_loss: 1.1263 - val_accuracy: 0.6234
Epoch 9/10
1563/1563 [==============================] - 3s 2ms/step - loss: 0.7567 - accuracy: 0.7343 - val_loss: 1.1237 - val_accuracy: 0.6260
Epoch 10/10
1563/1563 [==============================] - 3s 2ms/step - loss: 0.7154 - accuracy: 0.7509 - val_loss: 1.1630 - val_accuracy: 0.6156
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

![image](https://user-images.githubusercontent.com/43855029/192882693-1108d2ba-0f34-45ee-9168-9e0f7fd534d6.png)

### Discussions:
- We can see the accuracy increased from 0.44 to 0.75 for training and 0.6 for testing which is the great improvement
- The deeper the network the better?

### Save the CNN model for CIFAR10

```python
model.save('model1_CNN_CIFAR10.keras')
```

### Improving the model?

The model can be more accurate if you increase the size of it.

In the revised version, let increase the size of hidden layers:

```python
model = Sequential()
model.add(Conv2D(8, (3, 3), strides=(1, 1), activation='relu', input_shape=(32, 32, 3)))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(512, (3, 3), activation='relu'))

model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))

model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))

# compile model
model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])               
```

```python
model.summary()
```

```
Model: "sequential_11"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_30 (Conv2D)          (None, 30, 30, 8)         224       
                                                                 
 max_pooling2d_20 (MaxPoolin  (None, 15, 15, 8)        0         
 g2D)                                                            
                                                                 
 conv2d_31 (Conv2D)          (None, 13, 13, 512)       37376     
                                                                 
 max_pooling2d_21 (MaxPoolin  (None, 6, 6, 512)        0         
 g2D)                                                            
                                                                 
 conv2d_32 (Conv2D)          (None, 4, 4, 256)         1179904   
                                                                 
 flatten_10 (Flatten)        (None, 4096)              0         
                                                                 
 dense_19 (Dense)            (None, 100)               409700    
                                                                 
 dense_20 (Dense)            (None, 10)                1010      
                                                                 
=================================================================
Total params: 1,628,214
Trainable params: 1,628,214
Non-trainable params: 0
```

Now we can see that the number of parameters increased from 226k to 1.6 million!

Let's train the model

```python
model_CNN = model.fit(X_train, y_train, epochs=10, 
                    validation_data=(X_test, y_test))
```

```
Epoch 1/10
1563/1563 [==============================] - 6s 4ms/step - loss: 1.4947 - accuracy: 0.4533 - val_loss: 1.1870 - val_accuracy: 0.5721
Epoch 2/10
1563/1563 [==============================] - 5s 3ms/step - loss: 1.1082 - accuracy: 0.6083 - val_loss: 1.0441 - val_accuracy: 0.6349
Epoch 3/10
1563/1563 [==============================] - 5s 3ms/step - loss: 0.9404 - accuracy: 0.6690 - val_loss: 0.9733 - val_accuracy: 0.6665
Epoch 4/10
1563/1563 [==============================] - 5s 4ms/step - loss: 0.8229 - accuracy: 0.7118 - val_loss: 0.8952 - val_accuracy: 0.6976
Epoch 5/10
1563/1563 [==============================] - 5s 3ms/step - loss: 0.7262 - accuracy: 0.7439 - val_loss: 0.9003 - val_accuracy: 0.6946
Epoch 6/10
1563/1563 [==============================] - 5s 3ms/step - loss: 0.6406 - accuracy: 0.7754 - val_loss: 0.8999 - val_accuracy: 0.6979
Epoch 7/10
1563/1563 [==============================] - 5s 3ms/step - loss: 0.5609 - accuracy: 0.8028 - val_loss: 0.9167 - val_accuracy: 0.6990
Epoch 8/10
1563/1563 [==============================] - 5s 3ms/step - loss: 0.4875 - accuracy: 0.8279 - val_loss: 0.9412 - val_accuracy: 0.7164
Epoch 9/10
1563/1563 [==============================] - 5s 3ms/step - loss: 0.4146 - accuracy: 0.8536 - val_loss: 0.9965 - val_accuracy: 0.7086
Epoch 10/10
1563/1563 [==============================] - 5s 4ms/step - loss: 0.3499 - accuracy: 0.8756 - val_loss: 1.0582 - val_accuracy: 0.7045
```

By increasing the number of parameters from 226k to 1.6mil, we have the accuracy of training improved from 0.75 to 0.87 and testing 0.6 to 0.7!

Now, Let's save the model

```python
model.save('model2_CNN_CIFAR10.keras')
```

And validating with some testing data

```python
import numpy as np
predictions = model.predict(X_test)
ypreds = pd.Series(np.argmax(predictions, axis=1))

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
![image](https://user-images.githubusercontent.com/43855029/192893046-811a30b2-01d7-4b6f-bf71-7c3643af8c5d.png)

## Cluster Visualization using PCA
Principal component analysis (PCA) is a technique that transforms high-dimensions data into lower-dimensions while retaining as much information as possible.

![image](https://user-images.githubusercontent.com/43855029/193931906-792bb18b-3578-4433-b8ed-592a43e57611.png)

In this exercise, we implement the PCA technique to clusterize the CIFAR10 dataset

```python
from sklearn import decomposition
pca = decomposition.PCA(n_components=10)
pca_pred = pca.fit_transform(predictions)
```

Convert to DataFrame

```python
df_pca = pd.DataFrame(pca_pred,columns=['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10'])
df_pca['Cluster'] = y_test
df_pca['Cluster_name'] = ypreds.apply(lambda x:class_names[x])
```

Visualize the Principal Components:

```python
import seaborn as sns
df_pca_var = pd.DataFrame({'var':pca.explained_variance_ratio_,
             'PC':['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10']})
sns.barplot(x='PC',y="var", 
           data=df_pca_var);
```

![image](https://user-images.githubusercontent.com/43855029/193932226-03d6a413-2c23-4165-9e6a-99368b5c533c.png)


```python
sns.lmplot( x="PC1", y="PC2",
  data=df_pca, 
  fit_reg=False, 
  hue='Cluster_name', # color by cluster
  legend=True,
  scatter_kws={"s": 5}, # specify the point size
  height=8)           
```  

![image](https://user-images.githubusercontent.com/43855029/193932338-a57ba2ea-7bd7-48df-8938-55f706f7679d.png)

We can also visualize using images:

```python
tx, ty = df_pca['PC1'],df_pca['PC2']
tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))
```

```python
from PIL import Image
width = 4000
height = 3000
max_dim = 100
full_image_PCA = Image.new('RGB', (width, height))
for idx, x in enumerate(X_test):
    tile = Image.fromarray(np.uint8(x * 255))
    rs = max(1, tile.width / max_dim, tile.height / max_dim)
    tile = tile.resize((int(tile.width / rs),
                        int(tile.height / rs)),
                       Image.Resampling.LANCZOS)
    full_image.paste(tile, (int((width-max_dim) * tx[idx]),
                            int((height-max_dim) * ty[idx])))
full_image_PCA   
```

![image](https://user-images.githubusercontent.com/43855029/193932551-6d98f0e2-9382-4e8d-bf22-0d577e400822.png)

## Clusters Visualization using t-SNE

t-Distributed Stochastic Neighbor Embedding (t-SNE) is an unsupervised, non-linear technique primarily used for data exploration and visualizing high-dimensional data.

In simpler terms, t-SNE gives you a feel or intuition of how the data is arranged in a high-dimensional space. It was developed by Laurens van der Maatens and Geoffrey Hinton in 2008.

### t-SNE vs PCA?
- PCA developed in 1933 whilst t-SNE developed in 2008
- PCA is linear dimension reduction technique, that seeks to maximize variance and preserves large pairwise distances. In other words, things that are different end up far apart. This can lead to poor visualization especially when dealing with non-linear manifold structures.
- SNE differs from PCA by preserving only small pairwise distances or local similarities whereas PCA is concerned with preserving large pairwise distances to maximize variance.

[More Information](https://towardsdatascience.com/an-introduction-to-t-sne-with-python-example-5a3a293108d1)

```python
from sklearn.manifold import TSNE
model_tsne = TSNE(n_components=2)
tsne_pred = model_tsne.fit_transform(predictions)
```

Convert to DataFrame

```python
df_tsne = pd.DataFrame(tsne_pred,columns=['TSNE1','TSNE2'])
df_tsne['Cluster'] = ypreds
df_tsne['Cluster_name'] = ypreds.apply(lambda x:class_names[x])
```

Visualize with seaborn

```python
tsne1 = sns.lmplot( x="TSNE1", y="TSNE2",
  data=df_tsne, 
  fit_reg=False, 
  hue='Cluster_name', # color by cluster
  legend=True,
  scatter_kws={"s": 5}, # specify the point size
  height=8)
#tsne1.savefig('TSNE1.png')  
```  


Visualize with images

```python
tx, ty = df_tsne['TSNE1'], df_tsne['TSNE2']
tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))
```

```python
from PIL import Image
width = 4000
height = 3000
max_dim = 100
full_image_TSNE = Image.new('RGB', (width, height))
for idx, x in enumerate(X_test):
    tile = Image.fromarray(np.uint8(x * 255))
    rs = max(1, tile.width / max_dim, tile.height / max_dim)
    tile = tile.resize((int(tile.width / rs),
                        int(tile.height / rs)),
                       Image.Resampling.LANCZOS)
    full_image.paste(tile, (int((width-max_dim) * tx[idx]),
                            int((height-max_dim) * ty[idx])))
full_image_TSNE  
#full_image_TSNE.save("TSNE2.png")
```

![image](https://user-images.githubusercontent.com/43855029/193933220-f0d1a5bc-58cd-4221-98ff-7f4ed3202fbe.png)


## Loading a pre-trained model VGG16

- Now we can see that using the data it takes sometime to train the CNN model and it takes longer and longer if you have more and more dataset as well as increasing the number of parameters?

![image](https://user-images.githubusercontent.com/43855029/192888188-32540247-2921-4877-ac74-dbef217a09eb.png)

- Let's Rehearse from the above comparison for  [IMAGENET](https://image-net.org/) challange held every year, where many teams participated with different algorithms to classify 21k+ synset (things) of category from 14 millions+ images (updated number by Sep 2022). Link to download paper [here](https://www.image-net.org/static_files/papers/imagenet_cvpr09.pdf)

- The VGG16 - (Very Deep Convolutional Network) model secured first and second place in ImageNet2014 challange. In their research, the VGG16 used 16 layers deep and has about 144 million parameters.

![image](https://user-images.githubusercontent.com/43855029/192893697-ef218548-c975-4486-8312-66316b0ca440.png)


- So instead of training our own model, We can utilize the pre-trained VGG16 model for many other image recognigtion problem.

### Load VGG16 model from tensorflow

```python
from tensorflow.keras.applications import VGG16
  
# load the VGG16 network *pre-trained* on the ImageNet dataset
model = VGG16(weights="imagenet")
model.summary()
```

```
Model: "vgg16"
________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 224, 224, 3)]     0         
                                                                 
 block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792      
                                                                 
 block1_conv2 (Conv2D)       (None, 224, 224, 64)      36928     
                                                                 
 block1_pool (MaxPooling2D)  (None, 112, 112, 64)      0         
                                                                 
 block2_conv1 (Conv2D)       (None, 112, 112, 128)     73856     
                                                                 
 block2_conv2 (Conv2D)       (None, 112, 112, 128)     147584    
                                                                 
 block2_pool (MaxPooling2D)  (None, 56, 56, 128)       0         
                                                                 
 block3_conv1 (Conv2D)       (None, 56, 56, 256)       295168    
                                                                 
 block3_conv2 (Conv2D)       (None, 56, 56, 256)       590080    
                                                                 
 block3_conv3 (Conv2D)       (None, 56, 56, 256)       590080    
                                                                 
 block3_pool (MaxPooling2D)  (None, 28, 28, 256)       0         
                                                                 
 block4_conv1 (Conv2D)       (None, 28, 28, 512)       1180160   
                                                                 
 block4_conv2 (Conv2D)       (None, 28, 28, 512)       2359808   
                                                                 
 block4_conv3 (Conv2D)       (None, 28, 28, 512)       2359808   
                                                                 
 block4_pool (MaxPooling2D)  (None, 14, 14, 512)       0         
                                                                 
 block5_conv1 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                 
 block5_conv2 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                 
 block5_conv3 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                 
 block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0         
                                                                 
 flatten (Flatten)           (None, 25088)             0         
                                                                 
 fc1 (Dense)                 (None, 4096)              102764544 
                                                                 
 fc2 (Dense)                 (None, 4096)              16781312  
                                                                 
 predictions (Dense)         (None, 1000)              4097000   
                                                                 
=================================================================
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0
```

### Let's get some images:

```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

!wget https://cdn.britannica.com/29/150929-050-547070A1/lion-Kenya-Masai-Mara-National-Reserve.jpg

def show_image(image_path):
    image = mpimg.imread(image_path)
    print(image.shape)
    plt.imshow(image)
    
show_image("lion-Kenya-Masai-Mara-National-Reserve.jpg")   
```    

![image](https://user-images.githubusercontent.com/43855029/193933579-420e5448-a504-4ddd-b963-ee379d4139ab.png)


### Use the pretrained VGG16 model to process the image

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


### Prediction using VGG16
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

readable_prediction("lion-Kenya-Masai-Mara-National-Reserve.jpg")
```

![image](https://user-images.githubusercontent.com/43855029/193933781-8d57c620-f374-4f98-8fe2-2723848f706b.png)

```
(1085, 1600, 3)
Original image shape:  (1085, 1600, 3)
Processed image shape:  (1, 224, 224, 3)
1/1 [==============================] - 0s 315ms/step
Predicted: [[('n02129165', 'lion', 0.9831131), ('n02117135', 'hyena', 0.0046178186), ('n02486410', 'baboon', 0.0028358104)]]
```
### Other pre-trained model?

https://www.tensorflow.org/api_docs/python/tf/keras/applications
