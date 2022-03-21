---
title: "Tensorboard"
teaching: 20
exercises: 0
questions:
- "What is Tensorboard and How to use?"
objectives:
- "Open Tensorboard in M2"
keypoints:
- "Tensorboard"
---
#  Tensorboard

- TensorBoard is a visualization web app to get a better understanding of various parameters of the neural network model and its training metrics.
- Such visualizations are quite useful when you are doing experiments with neural network models and want to keep a close watch on the related metrics. 
- It is open-source and is a part of the Tensorflow group.

## Some of the useful things you can do with TensorBoard includes:

- Visualize metrics like accuracy and loss.
- Visualize model graph.
- Visualize histograms for weights and biases to understand how they change during training.
- Visualize data like text, image, and audio.
- Visualize embeddings in lower dimension space.

## Let setup the procedure to open Tensorboard with M2 HPC
Open Terminal in Open OD or regular terminal, make sure that you are in the same node with Jupyter Lab instance.

### Install Tensorboard

```python
pip install tensorboard
```

### Starting Tensorboard

In the previous episode, we already created the log files for CNN training under the same working folder (logs_CNN).
Now, let's load up the log files in the same directory:

```python
# First activate the working conda environment:
source activate ML_SKLN

# Second, change directory to where you have the log_CNN saved:
cd Workshop/SMU_DL

# Run the tensorboard with log data in the logs_CNN directory
tensorboard --logdir logs_CNN --host 0.0.0.0
```

The following information appears:

```python
2022-03-21 14:41:55.277397: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'libcudart.so.10.1'; dlerror: libcudart.so.10.1: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: :/users/tuev/Applications/lib:/users/tuev/Applications/lib
2022-03-21 14:41:55.277464: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
/users/tuev/.conda/envs/ML_SKLN/lib/python3.6/site-packages/tensorboard_data_server/bin/server: /lib64/libc.so.6: version `GLIBC_2.18' not found (required by /users/tuev/.conda/envs/ML_SKLN/lib/python3.6/site-packages/tensorboard_data_server/bin/server)
TensorBoard 2.8.0 at http://0.0.0.0:6017/ (Press CTRL+C to quit)
```



