Question Answering on SQuAD
===========================
This project implements models that train on the
[Stanford Question Answering Dataset](https://rajpurkar.github.io/SQuAD-explorer/)
(SQuAD). The SQuAD dataset is comprised of pairs of passages and questions
given in English text where the answer to the question is a span of text in the
passage. The goal of a model that trains on SQuAD is to predict the answer to
a given passage/question pair. The project's main site has examples of some of
the passages, questions, and answers, as well as a ranking for the
existing models.

Specifically, this project implements:
* [Match-LSTM](https://arxiv.org/abs/1608.07905): An early end-to-end neural
  network model that uses recurrent neural networks and an attention
  mechanism. This is more of a baseline with respect to other neural network
  models.
* [Rnet](aka.ms/rnet): A model that is similar to Match-LSTM, but adds several
  components to the model including a "gated" attention based mechanism, and
  a "self-matching" attention mechanism to match the passage against itself.
* [Mnemonic Reader](https://arxiv.org/abs/1705.02798): A model that uses a
  "feature-rich" encoder, iterative alignment of the passage and question,
  and a memory-based answer pointer layer. There is also a "reinforced" version
  of this model that uses reinforcement learning to fine-tune the model weights
  after initial training is done with gradient descent.

I primarily made this for my own education, but the code could be used as a
starting point for another project. Code is written in TensorFlow and uses
(optional) AWS S3 storage for model checkpointing and data storage.


Results
------------
|Model           | Dev Em            | Dev F1   |
| -------------- |:-----------------:| -------- |
|Match LSTM      | 59.4%             | 69.5%    |
|Rnet            | 61.4%             | 71.7%    |
|Mnemonic reader | 69.6%             | 78.6%    |

All results are for a single model rather than an ensemble.
I didn't train all models for a complete 10 epochs and there may be bugs or
unoptimized hyperparameters in my implementation.


Requirements
-------------
* [Python 3](https://www.python.org/downloads/)
* [Java 8 JRE](http://www.oracle.com/technetwork/java/javase/downloads/jre8-downloads-2133155.html).
  On Ubuntu 16.04 this can be installed with
  `sudo apt-get update && sudo apt-get install default-jre`
* Pip for Python 3 (`pip3`). On Ubuntu 16.04 this can be installed with
  `sudo apt-get update && sudo apt install python3-pip`
* If using GPU, install the
  [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) on your system

```
pip3 install --upgrade pip
pip3 install -r requirements.txt --user
```

Using AWS S3
--------------
In order to use AWS S3 for model checkpointing and data storage, you must set
up AWS credentials.
[This page](http://docs.aws.amazon.com/cli/latest/userguide/cli-config-files.html)
shows how to do it.

After your credentials are set up, you can enable S3 in the project by setting
the `use_s3` flag to `True` and setting `s3_bucket_name` to the name of your
S3 bucket.

```
f.DEFINE_boolean("use_s3", True, ...)
...
f.DEFINE_string("s3_bucket_name", "<YOUR S3 BUCKET HERE>",...)
```

How to run it
-------------
### Setup
Internet is required for setup. This takes 15-45 minutes. If you are using AWS
S3 storage, then this only needs to be done once, and the files will
automatically be downloaded if you run training or evaluation on another
machine.

```
python3 setup.py
```

### Training
If running on GPU, you will want to update the `num_gpus` flag to the number of
available GPUs you have on the system. For NVIDIA-based systems, you can tell
how many there are with a `nvidia-smi` command. Note that flags can be
updated in code like below or from the command line like this: `--flag=value`.

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
To run evaluation, use a command from below. This will evaluate the model
on the Dev dataset and print out the exact match (em) and f1 scores.
In addition, if the `visualize_evaluated_results` flag is `true`, then
the passsages, questions, ground truth spans, and spans predicted by the
model will be written to output files specified in the `evaluation_dir`
flag.

```
python3 evaluate_local.py # for local training
# OR
python3 evaluate_remote.py # for remote training - convenience (see below)
```

### Visualizing training
You can visualize the model loss, gradients, exact match, and f1 scores as the
model trains by using TensorBoard at the top level directory of this
repository.
```
tensorboard --logdir=log
```

Making changes
--------------

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

#### My workflow

I use an EC2 instance to train or test my models, with these steps:
1. Update the code.
2. Start a GPU EC2 instance. `p2.xlarge` and `p3.2xlarge` instances work OK.
   I load an AMI that has the NVIDIA Toolkit and python modules already
   installed.
3. Zip the files (`./zip.sh` from the top level directory).
4. Copy the files to EC2 instance:
   `scp -i /path/to/key.pem files.tar.gz ubuntu@<INSTANCE DNS>:~/`.
   This is a small amount of data (just the python files) since model weights
   and training data are copied from S3.
5. SSH to the instance: `ssh -i /path/to/key.pem ubuntu@<INSTANCE DNS>`
6. Unzip the files: `tar -xzvf files.tar.gz`
7. Run the "remote" training or evaluation steps from above in a `tmux`
   terminal.
8. If training, start another `tmux` terminal for tensorboard. You may need to
   open the port tensorboard is running on.
