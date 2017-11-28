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
  after initial training is done with gradient descent, although my
  implementation does not include that.

I primarily made this for my own education, but the code could be used as a
starting point for another project. Code is written in TensorFlow and uses
(optional) AWS S3 storage for model checkpointing and data storage.


Results
------------
|Model           | Dev Em            | Dev F1   |
| -------------- |:-----------------:| -------- |
|Match LSTM      | 59.4%             | 69.5%    |
|Rnet            | 61.4%             | 71.7%    |
|Mnemonic reader | 70.5%             | 79.8%    |

All results are for a single model rather than an ensemble.
I didn't train all models for the same duration and there may be bugs or
unoptimized hyperparameters in my implementation.


Requirements
-------------
* [Python 3](https://www.python.org/downloads/)
* [spaCy](https://spacy.io/) and the "en" model
* Tensorflow >= 1.3

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
```
python3 setup.py
```

### Training
```
python3 train_local.py --num_gpus=<NUMBER OF GPUS>
```

### Evaluation
The following command will evaluate the model
on the Dev dataset and print out the exact match (em) and f1 scores.
In addition, if the `visualize_evaluated_results` flag is `true`, then
the passsages, questions, ground truth spans, and spans predicted by the
model will be written to output files specified in the `evaluation_dir`
flag.

```
python3 evaluate_local.py --num_gpus=<NUMBER OF GPUS>
```

### Visualizing training
You can visualize the model loss, gradients, exact match, and f1 scores as the
model trains by using TensorBoard at the top level directory of this
repository.
```
tensorboard --logdir=log
```
