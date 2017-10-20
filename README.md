Question Answering on SQuAD
===========================
This project implements a model that trains on the Stanford Question Answering Dataset. I primarily made this for my own education, but the code could be used as a starting point for a research project that improves the results, or uses the model for a real world application. The Match LSTM in this implementation achieves an exact match score of ~59.5% and f1 score of ~69.5% after 10-20 hours of training. It is based on the model described in [Wang et. al](https://arxiv.org/pdf/1608.07905.pdf) with a few other ideas borrowed from other papers (including character-level embeddings, self-matching and gated attention, word-in-question and word-in-context features).

Code is written in TensorFlow and uses AWS S3 storage for model checkpointing and data storage (optional).

Requirements
-------------
* [Python 3](https://www.python.org/downloads/)
* [Java 8 JRE](http://www.oracle.com/technetwork/java/javase/downloads/jre8-downloads-2133155.html). On Ubuntu 16.04 this can be installed with `sudo apt-get update && sudo apt-get install default-jre`
* Pip for Python 3 (`pip3`). On Ubuntu 16.04 this can be installed with `sudo apt-get update && sudo apt install python3-pip`
* If using GPU, install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) on your system

```
pip3 install --upgrade pip
pip3 install -r requirements.txt --user
```

Using AWS S3
--------------
In order to use AWS S3 for model checkpointing and data storage, you must set up AWS credentials. [This page](http://docs.aws.amazon.com/cli/latest/userguide/cli-config-files.html) shows how to do it.

After your credentials are set up, you can enable S3 in the project by setting the `use_s3` flag to `True` and setting `s3_bucket_name` to the name of your S3 bucket.

```
f.DEFINE_boolean("use_s3", True, ...)
...
f.DEFINE_string("s3_bucket_name", "<YOUR S3 BUCKET HERE>",...)
```

How to run it
-------------
### Setup
Internet is required for setup. This takes 15-45 minutes. If you are using AWS S3 storage, then this only needs to be done once, and the files will automatically be downloaded if you run training or evaluation on another machine.

```
python3 setup.py
```

### Training
If running on GPU, you will want to update the `num_gpus` flag to the number of available GPUs you have on the system. For NVIDIA-based systems, you can tell how many there are with a `nvidia-smi` command.

```
# In flags.py
f.DEFINE_string("num_gpus", <NUMBER OF GPUS>, ..)
```

To run training:

```
python3 train_local.py # for local training
# OR
python3 train_remote.py # for remote training - convenience (see below)
```

### Evaluation
```
python3 evaluate_local.py # for local training
# OR
python3 evaluate_remote.py # for remote training - convenience (see below)
```

### Visualizing training
You can visualize the model loss, gradients, exact match, and f1 scores as the model trains by using TensorBoard at the top level directory of this repository.
```
tensorboard --logdir=log
```

Making changes
--------------

#### My workflow

I use an EC2 instance to train or test my models, with these steps:
1. Update the code.
2. Start a GPU EC2 instance. I recommend a spot request for a `p2.2xlarge` instance. I load an AMI that has the NVIDIA Toolkit and python modules already installed.
3. Zip the files (`./zip.sh` from the top level directory).
4. Copy the files to EC2 instance: `scp -i /path/to/key.pem files.tar.gz ubuntu@<INSTANCE DNS>:~/`
5. SSH to the instance: `ssh -i /path/to/key.pem ubuntu@<INSTANCE DNS>`
6. Unzip the files: `tar -xzvf files.tar.gz`
7. Run the "remote" training or evaluation steps from above in a `tmux` terminal.
8. If training, start another `tmux` terminal for tensorboard (make sure your tensorboard port is open).

#### Directory structure

    datasets/
        Defines SQuAD and debug datasets
    model/
        Defines models such as Match LSTM and Rnet to run on SQuAD
    preprocessing/
        Code used to download and create training/dev data, and (optionally) upload to S3
    test/
        Debugging code
    train/
        Code used to train and evaluate models
    flags.py
        Common model and training parameters for the program
    setup.py
        Runs preprocessing to create training data
    train_local.py
        Runs training locally
    train_remote.py
        Same as train_local, but overrides some flags to run on EC2 GPUs - mostly for convenience
    evaluate_local.py
        Evaluates the model on the Dev dataset
    evaluate_remote.py
        Same as evaluate_local.py, but overrides some flags to run on EC2 GPUs - mostly for convenience

