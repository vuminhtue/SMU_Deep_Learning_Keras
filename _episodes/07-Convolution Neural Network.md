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


### Prepare the data
Here we use iris data from ANN and Classification espisode in our previous Machine Learning class with sklearn:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd

iris = load_iris()
X = iris.data
y = iris.target
```


