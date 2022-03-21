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
# First, change directory to where you have the log_CNN saved:
cd Workshop/SMU_DL

# Run the tensorboard with log data in the logs_CNN directory
tensorboard --logdir logs_CNN
```


