---
title: "Recap on ANN"
teaching: 40
exercises: 0
questions:
- "Basic of ANN"
objectives:
- "ANN"
keypoints:
- "ANN"
---

## Recap on Artificial Neural Network (ANN)
Previous ANN in Machine Learning lecture can be found here: https://vuminhtue.github.io/Machine-Learning-Python/10-Neural-Network/index.html

![image](https://user-images.githubusercontent.com/43855029/114472746-da188c00-9bc0-11eb-913c-9dcd14f872ac.png)

Biological Neural Network

![image](https://user-images.githubusercontent.com/43855029/114472756-dd137c80-9bc0-11eb-863d-7c4d054efa89.png)

Machine Learning Neural Network (normally consists of 1 hidden layer: Shallow Network)

![image](https://user-images.githubusercontent.com/43855029/119180080-cf61da00-ba3d-11eb-9ad4-26c159a470be.png)

Deep Neural Network (multiple hidden layers)

### Forward Propagation (Feed-forward)
- The process that data passes through multiple layers of neuron from input layer to output layers.
- The image below demonstrates that data from input layer passing through 1 neuron by connection through the dendrite with specific weight for each connection. 
- The bias is also added to adjust the output from neuron
- Output of neuron is converted using activation function in order to map with the output layer.
- Since there is only 1 way direction of the data, so this method is called forward propagation
![image](https://user-images.githubusercontent.com/43855029/114472776-e997d500-9bc0-11eb-9f70-450389c912df.png)

### Backpropgation
- Forward Propagation is fast but it has limitation that the weights and bias cannot be changed.
- In order to overcome this limitation, Backpropagation calculate the error between output and observered data and propagate the error back into the network and update the weights and biases
- The Backpropagation process is repeated until number of iteration/epochs is reached or the error between output and observed is within the acceptable threshold.

### Activation Functions
Typical activation functions in Neural Network:
- Step Function (Binary Function)
- Linear Function
- Sigmoid Function*
- Hyperbolic Tangent Function (Tanh)*
- Rectified Linear Unit Function (ReLU)*

"*" Most popular functions

(1) Sigmoid Function

![image](https://user-images.githubusercontent.com/43855029/119183889-af80e500-ba42-11eb-923f-8314b3f88734.png)

- One of the widely used activation function in the hidden layer of NN
- However, it is flat with |z|>3, therefore it might lead to "vanishing gradient" in Backpropagation approach, that slowdown the optimization of NN in Deep Neural Network.
- Sigmoid function converts the output to range (0, 1) so it is not symmetric around the origin. All values are positive.

(2) Hyperbolic Tangent Function 

![image](https://user-images.githubusercontent.com/43855029/119186714-777ba100-ba46-11eb-8e8f-f82ce0954a91.png)

- Tanh is quite similar to Sigmoid but it is symmetric around the origin
- However, it also flat with |z|>3 and also lead to "vanishing gradient" problem in Deep Neural Network

(3) ReLU Function

![image](https://user-images.githubusercontent.com/43855029/119186990-b1e53e00-ba46-11eb-8f1c-637b546e62e8.png)

- The most widely used Activation Function in Deep Neural Network
- It is nonlinear
- It does not activate all neuron at the same time: If the input is negative, the neuron is not activated
- Therefore, it overcomes the "vanishing gradient" problem
 



