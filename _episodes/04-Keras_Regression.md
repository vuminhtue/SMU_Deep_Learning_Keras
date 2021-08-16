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

#### Layers in Deep Learning

![image](https://user-images.githubusercontent.com/43855029/129512909-a8eaa507-4869-4956-8bf9-ac6a5d6b4cd5.png)

- Input layer
- Dense (fully connected) layers
- Recurrent layer
- Convolution layer
- Other layers

#### Sequential

- A **Sequential** model is appropriate for a **plain stack of layers** where each layer has exactly one input tensor and one output tensor.
- More information can be found [here](https://keras.io/guides/sequential_model/)

#### Dense:

![image](https://user-images.githubusercontent.com/43855029/129509811-8b951430-dc5f-47d4-a31b-a12b6edade12.png)

**Dense** is a fully connected layer;

**Dense** implements the operation: output = activation(dot(input, kernel) + bias); where:
- activation is the element-wise activation function passed as the activation argument,
- kernel is a weights matrix created by the layer,
- bias is a bias vector created by the layer (only applicable if use_bias is True).
- More information on Dense can be found [here](https://keras.io/api/layers/core_layers/dense)

### Create a Sequential model with 3 layers:

```python
# Create a Sequential model
model = Sequential()
# Create a first hidden layer, the input for the first hidden layer is input layer which has 3 variables:
model.add(Dense(50, activation='relu', input_shape=(3,)))
# Create a second hidden layer
model.add(Dense(50, activation='relu'))
# Create an output layer with only 1 variable:
model.add(Dense(1), activation = 'relu')
```

### Optimal activation function?
#### For hidden layers:

![image](https://user-images.githubusercontent.com/43855029/129512679-34174dd4-8b79-4625-96d9-c85e5ea95c48.png)

#### For output layers:

![image](https://user-images.githubusercontent.com/43855029/129512553-17bf8d4e-5ed4-4180-aaa7-d180c2d093c0.png)

More information can be found [here](https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/)

