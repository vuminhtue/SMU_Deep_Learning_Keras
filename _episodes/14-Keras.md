---
title: "Unsupervised Learning"
teaching: 20
exercises: 0
questions:
- "Keras"
objectives:
- "Learn to use Keras"
keypoints:
- "Kras"
---

#Keras
```python
def keras_regression_model():
    # Create model
    model = Sequential()
    # Create one hidden layer of 10 nodes and ReLU activation function
    model.add(Dense(10,activation='relu',input_shape=(n_col,)))
    model.add(Dense(10,activation='relu'))
    # Create one ouput layer
    model.add(Dense(1))
    
    # Compile model with adam optimizer and mean squared error as the loss function
    model.compile(optimizer='adam',loss='mean_squared_error')
    return model
```

```python
import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn import metrics

concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')
concrete_data.head()

X = concrete_data.drop("Strength",axis=1)
y = concrete_data["Strength"]
```

```python
mse1=[]
for i in range(50):

    #Step 1. Prepare data
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
    n_col = X.shape[1]

    # Build Neural Network with Keras library
    # Step 2. Train the model on the training data using 50 epochs.
    model = keras_regression_model()
    model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=50,verbose=2)

    # Step 3. Evaluate model with mean_squared_error
    y_pred = model.predict(X_test)

    mse1.append(metrics.mean_squared_error(y_test,y_pred))
```
