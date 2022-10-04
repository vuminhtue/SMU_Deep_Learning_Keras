---
title: "Recurrent Neural Network for Stock forecasting"
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

# Hands-on exercise on application of LSTM in Stock forecasting
Here, we will access Keras LSTM to forecast the closing stock price given the information of its volume

### Stock data
- Here we use a sample Microsoft stock downloaded from yahoo finance
- Data consists of daily stock from certain time frame with Closing, Opening, Volume of stocks
- Temporal timeframe 4 years since 01/01/2018

- Input variables: Daily closing stock price and its volume
- Output or target variable: Closing stock price at later date

### Objective
- Using data from previous 30 days, forecast closing price the next day

#### Library requested:

Other than tensorflow.keras installed in previous workshop, we need additional library:

```python
!pip install pandas_datareader yfinance
```

#### Loading library:

```python
import numpy as np
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt

# For reading stock data from yahoo finance, install if needed
from pandas_datareader.data import DataReader
import yfinance as yf

# For time stamps
from datetime import datetime

import tensorflow.keras as keras
```

#### Define stock, starting date and import:

```python
# The stock we use here is Microsoft (MSFT) but feel free to use any other ticker.
stock = "MSFT"
start_date = '2018-01-01'
end_date = datetime.now()
```

```python
# Get the stock quote starting at certain date until current time
df = DataReader(stock, data_source='yahoo', start=start_date, end=end_date)
```

```python
	         High	      Low	      Open	      Close	      Volume	   Adj Close
Date						
2018-01-02	86.309998	85.500000	86.129997	85.949997	22483800.0	81.530235
2018-01-03	86.510002	85.970001	86.059998	86.349998	26061400.0	81.909653
2018-01-04	87.660004	86.570000	86.589996	87.110001	21912000.0	82.630577
2018-01-05	88.410004	87.430000	87.660004	88.190002	23407100.0	83.655052
2018-01-08	88.580002	87.599998	88.199997	88.279999	22113000.0	83.740417
...	...	...	...	...	...	...
2022-04-01	310.130005	305.540009	309.369995	309.420013	27085100.0	309.420013
2022-04-04	315.109985	309.709991	310.089996	314.970001	24289600.0	314.970001
2022-04-05	314.869995	309.869995	313.269989	310.880005	23156700.0	310.880005
2022-04-06	307.000000	296.709991	305.190002	299.500000	40058900.0	299.500000
2022-04-07	303.649902	296.359985	296.660004	302.839996	22547999.0	302.839996
1075 rows × 6 columns
```

Plotting heatmap:

```python
sns.pairplot(df)
```

![image](https://user-images.githubusercontent.com/43855029/162282013-59696172-dec1-4926-989e-6ee7cd90c8cd.png)

We can see that Closing price has very good correlation with High/Low/Open so in order to avoid colinearity, we choose only Closing Price and Volume as input for our RNN:

```python
fig = plt.figure(figsize=(12,12))
ax1 = fig.add_subplot(111)
ax1.plot(df['Close'],color='red',label="Closing price")
ax2 = ax1.twinx()
ax2.bar(df.index,df['Volume']/10**6,label="Volume")
ax2.grid(False)
plt.title("Daily Closing stock price and its volume of " + stock)
plt.xlabel("Date")
ax1.set_ylabel("Closing price (US$)")
ax2.set_ylabel("Volume (million)")
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
plt.show()
```

![image](https://user-images.githubusercontent.com/43855029/162282261-de7ef733-97c4-43cd-8e70-3ddefba7dc89.png)

```python
# Create a new dataframe with only the 'Close' and 'Volume' 
df = df.filter(['Close','Volume'])
```

## Setting up parameters

- Data was collected at daily interval.
- Using historical data of 30 days in the past: = **30 data points**
- To forecast stock the subsequent day = **1 data points**
- Data partition to **80% training** and **20% testing** in order of time
- For Neural Network, following parameters are pre-selected:
   - **Learning rate** = 0.001
   - **Batch size** = 24 (Batch size is the number of samples that usually pass through the neural network at one time)
   - **Epoch** = 50 (Epoch is the number of times that the entire dataset pass through the neural network)

```python
split_fraction = 0.8
train_split = int(split_fraction * int(df.shape[0]))

step = 1 
past = 30

batch_size = 24
epochs = 50
learning_rate=0.001
```

## Standardized data:

As input data has different range, so there would be the need for standardization

```python
from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler(feature_range=(0,1))
scaled_features = pd.DataFrame(scale.fit_transform(df))
scaled_features.columns = df.columns
scaled_features.index = df.index

# The split of training and testing data must follow the sequential
train_data = scaled_features[0:train_split]
test_data =  scaled_features[train_split:]

train_data.head()
```
```python
	         Close	   Volume
Date		
2018-01-02	0.003642	0.131973
2018-01-03	0.005192	0.166961
2018-01-04	0.008136	0.126381
2018-01-05	0.012321	0.141002
2018-01-08	0.012669	0.128346
```


## Data partitioning
### Selecting input/output for training/testing dataset:

![image](https://user-images.githubusercontent.com/43855029/162282565-602f356b-64ec-49b8-9e41-2b39e579fcf3.png)

### Training

```python
start_ytrain = past
end_ytrain = train_split + start_ytrain

x_train = train_data
y_train = scaled_features[start_ytrain:end_ytrain]["Close"]

sequence_length = int(past/step)
```

### Testing

```python
start_ytest = end_ytrain
end_ytest = len(test_data) - past

x_test = test_data.iloc[:end_ytest,:]
y_test = scaled_features.iloc[start_ytest:]["Close"]
```

### Validation

```python

valid_data = scaled_features.iloc[train_split - past: , :]
valid_data = np.array(valid_data)

x_valid = []
for i in range(past, len(valid_data)):
    x_valid.append(valid_data[i-past:i,:])

x_valid = np.array(x_valid)

```

### Using Keras to split training/testing data to different batch:
Here, we utilize the preprocessing time series feature of keras to split training/testing data into different batch:

### Training

```python
dataset_train = keras.preprocessing.timeseries_dataset_from_array(
    x_train,
    y_train,
    sequence_length=sequence_length,
    sampling_rate = step,
    batch_size=batch_size,
)
```

```python
for batch in dataset_train.take(1):
    inputs, targets = batch

print("Input shape:", inputs.numpy().shape)
print("Target shape:", targets.numpy().shape)
```


### Testing

```python
dataset_test = keras.preprocessing.timeseries_dataset_from_array(
    x_test,
    y_test,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size
)
```

```python
for batch in dataset_test.take(1):
    inputs_test, targets_test = batch

print("Input shape:", inputs_test.numpy().shape)
print("Target shape:", targets_test.numpy().shape)
```

#### Build Deep learning model with LSTM framework:

```python
inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
lstm_out = keras.layers.LSTM(64, activation="relu")(inputs)
outputs = keras.layers.Dense(1)(lstm_out)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
model.summary()
```

```
Model: "functional_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 30, 2)]           0         
_________________________________________________________________
lstm (LSTM)                  (None, 64)                17152     
_________________________________________________________________
dense (Dense)                (None, 1)                 65        
=================================================================
Total params: 17,217
Trainable params: 17,217
Non-trainable params: 0
_________________________________________________________________
```

#### Train the LSTM model and vaidate with testing data set:

```python
history = model.fit(
    dataset_train,
    epochs=epochs,
    validation_data=dataset_test,
    verbose=2
)
```

### Visualize the Training & Testing loss with 50 different epoches?

```python
def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

visualize_loss(history, "Training and Validation Loss")
```

![image](https://user-images.githubusercontent.com/43855029/162282980-0f73ac22-9f60-4374-acc4-e6549a8fbfa5.png)

### Evaluation with Forecasting

```python
y_pred = model.predict(x_valid)

#Create transformation function to rescale back to original
scaley = MinMaxScaler(feature_range=(0,1))
scaley.fit_transform(pd.DataFrame(df.iloc[:,0]))
y_pred = scaley.inverse_transform(y_pred)
```

#### Plot the data

```python
train = df[:train_split]
valid = df[train_split:]
valid['Predictions'] = y_pred
# Visualize the data
plt.figure(figsize=(16,6))
plt.title(stock + ' price prediction')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
```

![image](https://user-images.githubusercontent.com/43855029/162283231-4835202b-7c0b-496d-a9ec-c5c4a236166c.png)

#### For batch predictions:

```python
def show_plot(plot_data, title):
    labels = ["History", "True Future", "Model Prediction"]
    marker = [".-", "rx", "go"]
    time_steps = list(range(-(plot_data[0].shape[0]), 0))
    

    plt.title(title)
    for i, val in enumerate(plot_data):
        if i:
            plt.plot(0, plot_data[i], marker[i], markersize=10, label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlabel("Time-Step")
    plt.ylabel("Closing cost US$()")
    plt.show()
    return

# Scale the data
scaleT = MinMaxScaler(feature_range=(0,1))
scaleT.fit_transform(pd.DataFrame(df[:]))

for x, y in dataset_test.take(8):
    show_plot(
        [scaley.inverse_transform(pd.DataFrame(x[0][:, 0])),
         scaley.inverse_transform(pd.DataFrame(pd.Series(y[0].numpy()))),
         scaley.inverse_transform(pd.DataFrame(model.predict(x)[0]))],         
        "Single Step Prediction",
    )
```

![image](https://user-images.githubusercontent.com/43855029/162283319-f5e0671e-4726-4276-8d87-b5446e100605.png)

![image](https://user-images.githubusercontent.com/43855029/162283344-0022aa31-53f1-4d25-ab50-5573edf10b85.png)

## Save model

```python
model.save('Stock_prediction_MSFT.keras')
```

## Load the save model and predict tomorrow closing price for stock

```python
model1 = keras.models.load_model('Stock_prediction_MSFT.keras')
```

Load new data:

```python
stock = "MSFT"
start_date = '2018-01-01'
end_date = datetime.now()
df = DataReader(stock, data_source='yahoo', start=start_date, end=end_date)
```

Retain the last 30 data for prediction

```python
df = df.tail(30).filter(["Close","Volume"])
```

Scale data

```python
from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler(feature_range=(0,1))
X = pd.DataFrame(scale.fit_transform(df))

scaley= MinMaxScaler(feature_range=(0,1))
scaley.fit_transform(df[["Close"]])
```

Reshape input data:

```python
X = np.array(X).reshape(1,X.shape[0],X.shape[1])
```

Predict:

```python
y=model.predict(X)
print("Predicted Closing price tomorrow for ", stock, "is", scaley.inverse_transform(pd.DataFrame(y)), "$")

```

Show predicted plot:

```python
show_plot(
        [scaley.inverse_transform(pd.DataFrame(X[0])),
         scaley.inverse_transform(pd.DataFrame(y)),
         scaley.inverse_transform(pd.DataFrame(y))],         
        "Single Step Prediction")
```

![image](https://user-images.githubusercontent.com/43855029/162283651-f60346d5-5785-4aa3-b33d-de24fc055bdc.png)
