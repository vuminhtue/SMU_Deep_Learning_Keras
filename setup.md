---
title: Setup
---
Setup Jupyter kernel in ManeFrame M2's JupyterLab
---

### Install Tensorflow from existing skln conda environment (ML_SKLN kernel) used in previous Machine Learning workshop

Once open Jupyter Lab, open new terminal:

```
$ source activate ML_SKLN
$ pip install tensorflow==2.3.0
```

Check if keras is correctly installed:

```
>>> import tensorflow as tf
>>> from tensorflow import keras
>>> print(keras.__version__)
```

### Install Keras from new conda environment
Please follow this guideline to create a new conda environment and install **tensorflow** package.

Once open Jupyter Lab, open new terminal:

1. Create conda environment:

```
$ conda create -n my_keras python=3.8
```

2. Once done, activate the environment and install numpy, pandas, scikit-learn, matplotlib, seaborn and tensorflow


```
$ source activate my_keras
$ pip install numpy pandas scikit-learn seaborn
$ pip install tensorflow==2.3.0 tensorboard
$ conda install matplotlib 
```

=> Note: while using **my_keras** conda environment, if we are missing anything, we can always come back and update using **pip install**
or **conda install** method.

Check if keras is correctly installed:

```
>>> import tensorflow as tf
>>> from tensorflow import keras
>>> print(keras.__version__)
```

3. Last step: create Jupyter Hub kernel in order to work with Jupyter Notebook

```
$ conda install jupyter
$ python -m ipykernel install --user --name my_keras --display-name "DL_Keras"
```
4. Open Jupyter Lab and check if the kernel has been created?

{% include links.md %}

