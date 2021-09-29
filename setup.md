---
title: Setup
---
Setup Keras kernel in Palmetto's JupyterLab
---

### Install Keras from existing skln conda environment (ML_SKLN kernel)
Open terminal and activate the conda environment or using the consol with ML_SKLN kernel

```
$ pip install tensorflow==2.3.0
$ pip install keras==2.4.3
```

Check if keras is correctly installed:

```
>>> import keras
>>> print(keras.__version__)
```

### Install Keras from new conda environment
We will use Palmetto cluster for this workshop with Jupyter Lab.
Please follow this guideline to create a new conda environment and install **keras** package.

1. Open terminal (MobaXTerm for Windows OS/Terminal for MacOS & Linux platform)
2. Login to Palmetto login node: your_username@login001
3. Request for a compute node with simple configuration:


```
$ qsub -I -l select=1:ncpus=8:mem=32gb:interconnect=any,walltime=24:00:00
```

4. Load module:


```
$ module load anaconda3/2020.07-gcc/8.3.1
```

5. Create conda environment:


```
$ conda create -n my_keras python=3.8
```

6. Once done, activate the environment and install numpy, pandas, scikit-learn, matplotlib, seaborn and keras


```
$ source activate my_keras
$ pip install numpy pandas scikit-learn seaborn
$ pip install tensorflow==2.3.0
$ pip install keras==2.4.3
$ conda install matplotlib 
```

=> Note: while using **my_keras** conda environment, if we are missing anything, we can always come back and update using **pip install**
or **conda install** method.

Check if keras is correctly installed:

```
>>> import keras
>>> print(keras.__version__)
```

7. Last step: create Jupyter Hub kernel in order to work with Jupyter Notebook


```
$ conda install jupyter
$ python -m ipykernel install --user --name my_keras --display-name "DL_Keras"
```

8. Open Jupyter Lab in Palmetto, login and see if you have **DL_Keras** kernel created
https://www.palmetto.clemson.edu/jhub/hub/home

![image](https://user-images.githubusercontent.com/43855029/117865975-9159fe80-b264-11eb-94e7-bcbf17f1e55c.png)
{% include links.md %}

