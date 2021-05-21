---
title: "Training Deep Learning Regression model with Keras"
teaching: 20
exercises: 0
questions:
- "How to train a Deep Learning model using Regression method with Keras"
objectives:
- "Master Keras"
keypoints:
- "Regression training", "Keras"
---
## Train model using Logistic Regression
- Logistic regression is another technique borrowed by machine learning from the field of statistics. It is the go-to method for binary classification problems (problems with two class values).
- Typical binary classification: True/False, Yes/No, Pass/Fail, Spam/No Spam, Male/Female
- Unlike linear regression, the prediction for the output is transformed using a non-linear function called the logistic function.
- The standard logistic function has formulation:

![image](https://user-images.githubusercontent.com/43855029/114233181-f7dcbb80-994a-11eb-9c89-58d7802d6b49.png)

![image](https://user-images.githubusercontent.com/43855029/114233189-fb704280-994a-11eb-9019-8355f5337b37.png)

In this example, we use `breast cancer` data set built-in [sklearn data](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer).

This is a data set that classify breast cancer to `malignant` or `benign` based on different input data on the breast's measurement from 569 patients

Read in data:
```python
import pandas as pd
from sklearn.datasets import load_breast_cancer

datab = load_breast_cancer()
X = datab.data
y = datab.target
```
Standardize input data:
```python
from sklearn.preprocessing import scale
Xstd = pd.DataFrame(scale(X,axis=0, with_mean=True, with_std=True, copy=True))
```
Partitioning Data to train/test:
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xstd,y,train_size=0.6,random_state=123)
```
Train model using Logistic Regression
```python
from sklearn.Linear_Model import LogisticRegression
model_LogReg = LogisticRegression().fit(X_train, y_train)
y_pred = model_LogReg.predict(X_test)
```
Evaluate output with accurary level:
```python
from sklearn import metrics
metrics.accuracy_score(y_test,y_pred)
```
We retrieve the **accuracy = 0.99**
