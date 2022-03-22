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
- It is open-source and is a part of the Tensorflow version 2+

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
(ML_SKLN) [tuev@p014 SMU_DL]$ tensorboard --logdir logs_CNN --host 0.0.0.0
2022-03-21 14:41:55.277397: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'libcudart.so.10.1'; dlerror: libcudart.so.10.1: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: :/users/tuev/Applications/lib:/users/tuev/Applications/lib
2022-03-21 14:41:55.277464: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
/users/tuev/.conda/envs/ML_SKLN/lib/python3.6/site-packages/tensorboard_data_server/bin/server: /lib64/libc.so.6: version `GLIBC_2.18' not found (required by /users/tuev/.conda/envs/ML_SKLN/lib/python3.6/site-packages/tensorboard_data_server/bin/server)
TensorBoard 2.8.0 at http://0.0.0.0:6017/ (Press CTRL+C to quit)
```

From the above information, let's remember the node number (p014) and the port number (6017) and these numbers are different for different users.

```python
p014:6017
```

### Setting up Socket Proxy

#### For Macs OS/Linux

- Setup the SSH tunnel with port forwarding to M2.
- Open a terminal and run the following command with corresponding port number (6017 in this case)(this terminal must be kept opened for the socket proxy to be active):

```bash
$ ssh -D 6017 -C -q -N USERNAME@m2.smu.edu
```

- User need to authenticate themselve with password and DUO

#### For Window OS

- Install Moba XTerm (https://mobaxterm.mobatek.net)
- Open Moba XTerm and login M2:
- Click on Tunneling

![image](https://user-images.githubusercontent.com/43855029/159352196-45fc777e-bcc7-4e13-9679-65d2baf9d355.png)
 and select New SSH Tunnel:
 
 ![image](https://user-images.githubusercontent.com/43855029/159352273-0c670730-58de-4e9c-bb99-81e063857c8e.png)

- From the openning windows, select the following option:
    + Dynamic port forwarding (SOCKS proxy)
    + Forward port 6017
    + SSH Server: m2.smu.edu
    + Username: your own username
    + SSH Port 22

![image](https://user-images.githubusercontent.com/43855029/159352517-82bd36c2-b679-446c-8d30-7eecd4b28dbc.png)
 
- Press Save and the following window appears:

![image](https://user-images.githubusercontent.com/43855029/159352914-34075591-ba97-4c00-8dbe-e92e4c170348.png)

- Select the play button to start the service, The MobaXTerm will ask for your credential (including Duo) 
- Once accepting the Duo push, your Moba XTerm screen would look like this:
![image](https://user-images.githubusercontent.com/43855029/159353202-ba2fde99-2c13-40cc-9854-1ccf83af9f1b.png)

Leave the Window on and open Firefox:

### Setting up Firefox:
- Firefox should be enable to view website through the proxy.
- Once Firefox opened:
  + Windows: select "ALT+T+S" to open the setting tab.
  + Macs OS: Preferences\General
  
- Scroll all the way down, you  will see Network Settings. Click on Settings ...
![image](https://user-images.githubusercontent.com/43855029/159353682-0f295c34-2a88-4b0b-8d5a-b89c631bdd92.png)

- Fill in the following information:
+ Manual proxy configuration
+ SOCKS v5
+ Port 6017
+ Enable checkboxes for Proxy DNS and Enable DNS over HTTPS:
![image](https://user-images.githubusercontent.com/43855029/159354645-5ef3f328-d351-42f2-939a-4822d544babc.png)

- Click OK and open a new Tab in Firefox:
- Type in the address bar the node name and port number:

```python
p014:6017
```

The Tensorboard window appears:

![image](https://user-images.githubusercontent.com/43855029/159354281-a9fd7371-f14c-4c72-91ec-9daf4e4ac4dd.png)
