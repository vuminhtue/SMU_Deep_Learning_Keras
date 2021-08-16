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
## Convolution Neural Network - CNN

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


