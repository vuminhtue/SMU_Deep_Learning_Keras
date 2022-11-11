---
title: "Training Deep Learning Regression & Classiciation model with Keras Sequential"
teaching: 20
exercises: 0
questions:
- "How to train a Deep Learning model using Regression/Classication method with Keras"
objectives:
- "Master Keras"
keypoints:
- "Regression/Classification training, keras"
---
## Using Keras to solve a Regression Model

### Prepare the data
Here we use airquality data from ANN and Regression espisode in our previous Machine Learning class with sklearn:

```python
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

data_df = pd.DataFrame(pd.read_csv('https://raw.githubusercontent.com/vuminhtue/SMU_Deep_Learning_Keras/master/data/r_airquality.csv'))

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
X_test_scaled  = scaler.fit_transform(X_test)
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
model.add(Dense(50, activation='relu', input_shape=(3,)))
# Create a second hidden layer
model.add(Dense(40, activation='relu'))
# Create an output layer with only 1 variable:
model.add(Dense(1))
```

### Compile model

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

There are also many other **loss** function. 

The purpose of **loss** functions is to compute the quantity that a model should seek to minimize during training. Detail can be found [here](https://keras.io/api/losses/)

### Fit model

```python
model.fit(X_train_scaled, y_train, epochs=100, verbose=1,
               validation_data=(X_test_scaled,y_test))
```

Here: 
- epochs: the number of iteration 
- verbose: 'auto', 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. 'auto' defaults to 1 for most cases, but 2 when used with ParameterServerStrategy. Note that the progress bar is not particularly useful when logged to a file, so verbose=2 is recommended when not running interactively (eg, in a production environment).

There are alternative way if user do not want to split input data into training/testing:

```python
model.fit(X_scaled, y, validation_split=0.3, epochs=100, verbose=1)
```
 
### Evaluate model
Evaluate the testing set using given loss function
```python
results = model.evaluate(X_test_scaled, y_test, verbose=1)
print("test loss:", results)
```

### Predict output
```python
predictions = model.predict(X_test_scaled)
```

### Save & load keras model
```python
from keras.models import load_model

model.save('my_model.keras')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
model = load_model('my_model.keras')
```


## Using Keras to solve a Classification Model

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

### Apply One Hot Encoding to categorize the output:
- One Hot Encoding allows to represent the categorical data in a probabilistic way that is understandable by the machine.
- For example **cat, dog, deer** can be converted to **0, 1, 2** or **[1 0 0,0 1 0,0 0 1]**

```python
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
Y = enc.fit_transform(y[:, np.newaxis]).toarray()
```

### Split data to training/testing:

```python
X_train, X_test, y_train, y_test = train_test_split(X,Y,train_size=0.6,random_state=123)
```

### Scale the input data:

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
model.add(Dense(50, activation='relu', input_shape=(4,)))
# Create a second hidden layer
model.add(Dense(40, activation='relu'))
# Create an output layer with only 3 variables:
model.add(Dense(3,activation='softmax'))
```

### Compile model

```python
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics='accuracy')
```

### Fit model

```python
model.fit(X_train_scaled, y_train, epochs=100, verbose=1,
               validation_data=(X_test_scaled,y_test))
```
 
### Evaluate model
Evaluate the testing set using given loss function
```python
results = model.evaluate(X_test_scaled, y_test, verbose=1)
print("test loss, test acc:", results)
```

### Predict output
```python
predictions = model.predict(X_test_scaled)
```

### Inverse transform the One Hot Encoding
```python
Y_pred = enc.inverse_transform(predictions)
Y_test = enc.inverse_transform(y_test)
```

### Evaluate using accuracy score:

```python
import sklearn
sklearn.metrics.accuracy_score(Y_pred,Y_test)
```
