---
title: "Recurrent Neural Network for Timeseries forecasting"
teaching: 20
exercises: 0
questions:
- "How to train a RNN model with Keras"
objectives:
- "how to apply RNN model"
keypoints:
- "RNN, LSTM, keras"
---
# Recurrent Neural Networks
### Introduction
- RNNs are type of Deep Learning models with built-in feedback mechanism. 
- The output of a particular layer can be **re-fed** as the input in order to predict the output. 

![image](https://user-images.githubusercontent.com/43855029/132886824-3c84c35a-4c2d-4c0e-9529-16e3c4f0a0fe.png)
[source](https://www.simplilearn.com/tutorials/deep-learning-tutorial/rnn)

### Types of RNN

![image](https://user-images.githubusercontent.com/43855029/132903689-398ef108-660d-47ba-ae46-b783f203e307.png)

### Applications
- It is specifically designed for Sequential problem **Weather forecast, Stock forecast, Image captioning, Natural Language Processing, Speech/Voice Recognition**

### Some Disadvantages of RNN: 
- Computationally Expensive and large memory requested
- RNN is sensitive to changes in parameters such as they can grow exponentially (Exploding Gradient) or drop down to zero & stabalized (Vanishing Gradient).
- In order to resolve the cons of RNN, a method Long-Short Term Memory (LSTM) is proposed.

In this limited workshop, we only cover LSTM for timeseries forecast problem.

# Long-Short Term Memory model - LSTM

