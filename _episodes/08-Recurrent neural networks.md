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

![image](https://user-images.githubusercontent.com/43855029/132912049-167cf37e-66a0-4b54-8024-183ab7785398.png)

[source](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

- A look at detailed when we unroll the RNN loop:

![image](https://user-images.githubusercontent.com/43855029/132911838-0ce7eb99-fd60-44c7-b554-d176fdb45f8b.png)


### Types of RNN

![image](https://user-images.githubusercontent.com/43855029/132903689-398ef108-660d-47ba-ae46-b783f203e307.png)

### Applications
- It is specifically designed for Sequential problem **Weather forecast, Stock forecast, Image captioning, Natural Language Processing, Speech/Voice Recognition**

### Some Disadvantages of RNN: 
- Computationally Expensive and large memory requested
- RNN is sensitive to changes in parameters and having **Exploding Gradient** or **Vanishing Gradient**
- In order to resolve the gradient problem of RNN, a method Long-Short Term Memory (LSTM) is proposed.

In this limited workshop, we only cover LSTM for timeseries forecast problem.

# Long-Short Term Memory model - LSTM
### Introduction
- LSTMs are a special kind of RNN — capable of learning long-term dependencies by remembering information for long periods is the default behavior.
- They were introduced by Hochreiter & Schmidhuber (1997) and were refined and popularized by many people
- LSTMs are explicitly designed to avoid the long-term dependency problem.

### Comparison between traditional RNN and LSTM

![image](https://user-images.githubusercontent.com/43855029/132913273-1b7d4765-a8f2-4f2d-b3b9-6910d5d15807.png)

### Step by step walkthrought LSTM:
[Link](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

# Hands-on exercise on application of LSTM in temperature forecast
Here, we will access Keras LSTM to forecast temperature at site name Jena (Germany), given information of temperature and other climate variables.
The tutorial following the [keras website](https://keras.io/examples/timeseries/timeseries_weather_forecasting/), but rewritten in a simpler way for easy understanding.

### Climate Data
- Single station named Jena station in Germany.
- Data consists of 14 climate variables in every 10 minutes
- Temporal timeframe 8 year: 01/01/2009 - 12/31/2016
- Data description:


![image](https://user-images.githubusercontent.com/43855029/132914704-b2c7ee79-0c99-482a-abfd-cc4575dcfe1b.png)

- Input variables: all 14 climate variables
- Output or target variable: Temperature at later date

### Objective
- Using data from previous 5 days, forecast temperature in the next 12 hours

Here are the steps:

#### Loading library:

```python
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from numpy import array    
```

#### Loading Jena climate station data:

```python
df = pd.read_csv("jena_climate_2009_2016.csv")
```

