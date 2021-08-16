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

### Let's use Keras's Sequential model with Dense layers

```python
import keras
from keras.models import Sequential
from keras.layers import Dense
```

### Create a Sequential model with 3 layers:

```python
# Create a Sequential model
model = Sequential()
# Create a first hidden layer, the input for the first hidden layer is input layer which has 3 variables:
model.add(Dense(5, activation='relu', input_shape=(3,)))
# Create a second hidden layer
model.add(Dense(4, activation='relu'))
# Create an output layer with only 1 variable:
model.add(Dense(1), activation = 'relu')
```

### Compile and fit model

```python
model.compile(optimizer='adam', loss='mean_squared_error')
```

Here, **adam** optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments.

According to Kingma et al., 2014, the method is "computationally efficient, has little memory requirement, invariant to diagonal rescaling of gradients, and is well suited for problems that are large in terms of data/parameters".

More information on **adam** optimizer is [here](https://keras.io/api/optimizers/adam/)

In addition to **adam**, there are many other optimizer:
- [SGD](https://keras.io/api/optimizers/sgd)
- [RMSprop](https://keras.io/api/optimizers/rmsprop)
- [Adadelta](https://keras.io/api/optimizers/adadelta)
- [Adagrad](https://keras.io/api/optimizers/adagrad)
- [Adamax](https://keras.io/api/optimizers/adamax)
- [Nadam](https://keras.io/api/optimizers/nadam)
- [Ftrl](https://keras.io/api/optimizers/ftrl)

