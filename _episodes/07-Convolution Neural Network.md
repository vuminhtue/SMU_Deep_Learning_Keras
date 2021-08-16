---
title: "Training Convolution model with Keras"
teaching: 20
exercises: 0
questions:
- "How to train a CNN model with Keras"
objectives:
- "Master Keras"
keypoints:
- "CNN, keras"
---
## Convolutional Neural Network - CNN


- CNNs are one type of ANN which utilize the neuron, kernel, activation function.
- Inputs must be in images (or assumed to be images)
- Using Forward propagation technique with certain property to process it faster
- CNNs best for object detection, image classification, computer vision

### Architecture of CNNs


![image](https://user-images.githubusercontent.com/43855029/129609166-4589ffb7-eb89-403b-acfd-fe5c015b3cc7.png)
[Source](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)

- Consisting of Convolution Layers, Pooling Layers and fully connected Layer (Dense) before output layer
- A simple image can be simply flatten into 1D vector and driven through the regular fully connected NN. However, this requires lot of computational power if the image is large and has more color.

#### Convolutional Neural Network (CNN or ConvNet)

![image](https://user-images.githubusercontent.com/43855029/129623983-173558ba-45f5-4a42-972d-a6252f7695e0.png)

- CNN uses the Convolved Feature to reduce the image size by dot product with given kernel.
- The image reduction without losing features and easier to process for good prediction

![image](https://user-images.githubusercontent.com/43855029/129624312-db0f2ce1-4767-4a18-9a02-f5cee4c6cfe5.png)

- So for 3 channel RGB colors, the image size have been reduced:

![image](https://user-images.githubusercontent.com/43855029/129624564-96d6d7e4-6409-4775-ad9d-2bf133fa0396.png)

#### Pooling Layer
- Similar to the Convolutional Layer, the Pooling layer is responsible for reducing the spatial size of the Convolved Feature.
- This is to decrease the computational power required to process the data through dimensionality reduction
- Two types of Pooling: Max Pooling & Average Pooling.

![image](https://user-images.githubusercontent.com/43855029/129624678-75532145-0e90-48d5-9703-c8ee626aa7f4.png)

In which Max Pooling performs a lot better than Average Pooling.

#### Flatten Layer
- Once the images have passed through Convolution Layer and Pooling Layer, its size has been reduced greatly and ready for MLP training.
- The image is then flatten to a column vector and passed through feed-forward NN and BackPropagation applied to every iteration.
- Softmax activation function is applied to classified the multi-output
