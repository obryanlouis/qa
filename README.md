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
* [Match-LSTM](https://arxiv.org/abs/1608.07905)
* [Rnet](https://www.microsoft.com/en-us/research/publication/mrc/)
* [Mnemonic Reader](https://arxiv.org/abs/1705.02798)
* [Fusion Net](https://arxiv.org/abs/1711.07341)

I primarily made this for my own education, but the code could be used as a
starting point for another project. The models are written in TensorFlow and 
the project uses (optional) AWS S3 storage for model checkpointing and
data storage.


Results
------------
|Model                    | Dev Em            | Dev F1   | Details |
| ------------------------|:-----------------:| -------- |:------: |
|Match LSTM               | 59.4%             | 69.5%    |         |
|Rnet                     | 61.4%             | 71.7%    |         |
|Fusion Net               | 72.0%             | 81.2%    | Checkout [315c94979e1498707c4a1928d4c90db6a6d8f384](https://github.com/obryanlouis/qa/commit/315c94979e1498707c4a1928d4c90db6a6d8f384) `python3 train_local.py --model_type=fusion_net --input_dropout=0.6 --rnn_dropout=0.4 --dropout=0.4 --rnn_size=60 --batch_size=45 --use_token_reembedding=True` ~15 min/epoch over 2 1080 Ti GPUs        |
|Mnemonic reader (+ CoVe) | 72.5%             | 81.2%    | Checkout [b31a8e8ec1897c1eef8e80570cca19ea08b85467](https://github.com/obryanlouis/qa/commit/b31a8e8ec1897c1eef8e80570cca19ea08b85467) `python3 train_local.py --model_type=mnemonic_reader --rnn_size=60 --use_cove_vectors=True --dropout=0.3 --batch_size=50` training time ~5 hours over 2 1080 Ti GPUs, ~9.7 min/epoch     |

All results are for a single model rather than an ensemble.
I didn't train all models for the same duration and there may be bugs or
unoptimized hyperparameters in my implementation.


Requirements
-------------
* [Python 3](https://www.python.org/downloads/)
* [spaCy](https://spacy.io/) and the "en" model
* [Cove vectors](https://github.com/salesforce/cove) - You can skip this part
  but will probably need to manually remove any cove references in the setup.
  This also requires [pytorch](http://pytorch.org/).
* Tensorflow >= 1.3
* cuDNN 7 recommended, GPUs required

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
