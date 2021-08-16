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

#### Dense
