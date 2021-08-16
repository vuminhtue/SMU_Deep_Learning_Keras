---
title: "Training Deep Learning Regression model with Keras"
teaching: 20
exercises: 0
questions:
- "How to train a Deep Learning model using Regression method with Keras"
objectives:
- "Master Keras"
keypoints:
- "Regression training, keras"
---
## Using Keras to solve a Regression Model

### Prepare the data
Here we use airquality data from ANN and Regression espisode in our previous Machine Learning class with sklearn:

```python
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

data_df = pd.DataFrame(pd.read_csv('https://raw.githubusercontent.com/vuminhtue/Machine-Learning-Python/master/data/r_airquality.csv'))

imputer = KNNImputer(n_neighbors=2, weights="uniform")
data_knnimpute = pd.DataFrame(imputer.fit_transform(data_df))
data_knnimpute.columns = data_df.columns

X_train, X_test, y_train, y_test = train_test_split(data_knnimpute[['Temp','Wind','Solar.R']],
                                                    data_knnimpute['Ozone'],
                                                    train_size=0.6,random_state=123)
```

Now, we need to scale the input data:

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Let's use Keras

```python
import keras
from keras.models import Sequential
from keras.layers import Dense
```

#### Sequential

#### Dense:

![image](https://user-images.githubusercontent.com/43855029/129509811-8b951430-dc5f-47d4-a31b-a12b6edade12.png)


**Dense** implements the operation: output = activation(dot(input, kernel) + bias); where:
- activation is the element-wise activation function passed as the activation argument,
- kernel is a weights matrix created by the layer,
- bias is a bias vector created by the layer (only applicable if use_bias is True).
More information on Dense can be found [here](https://keras.io/api/layers/core_layers/dense)
