"""Define flags and provide a function to get options from the flags.
"""

import tensorflow as tf

f = tf.app.flags
f.DEFINE_integer("max_ctx_length", 300,
        "Max passage length to keep. Longer content will be trimmed.")
f.DEFINE_integer("max_qst_length", 60,
        "Max question length to keep. Longer content will be trimmed.")
f.DEFINE_string("model_type", "mnemonic_reader", "Type of model to train." +
        "The model types are in models/model_types.py")
f.DEFINE_string("experiment_name", "test",
        "Name of the experiment being run; different experiments will be " +
        "saved and loaded from different model files and can use different " +
        "model hyperparameters. If you use the same experiment name, then it" +
        "will be loaded from from the same model files (including using S3)")
f.DEFINE_string("checkpoint_dir", "checkpoint",
        "Directory to save model weights and metadata.")
f.DEFINE_float("learning_rate", 1e-3, "Initial learning rate.")
f.DEFINE_float("min_learning_rate", 1e-6,
        "Minimum learning rate, even after decay.")
f.DEFINE_float("learning_rate_decay", 0.7,
        "Learning rate to apply continuously over each epoch.")
f.DEFINE_string("data_dir", "data",
        "Directory with word embeddings, vocab, and training/dev data.")
f.DEFINE_string("evaluation_dir", "evaluation_results",
        "Directory that stores the results of evaluating models.")
f.DEFINE_boolean("visualize_evaluated_results", True,
        "Whether to write DEV contexts, questions, ground truths, and " +
        "predicted spans to a file when running evaluation.")
f.DEFINE_string("log_dir", "log", "Directory to log training summaries. " +
        "These summaries can be monitored with tensorboard.")
f.DEFINE_string("clear_logs_before_training", True,
        "Whether to clear the log directory before starting training.")
f.DEFINE_integer("log_every", 10, "Frequency to log loss and gradients.")
f.DEFINE_string("log_loss", True, "Whether to log loss summaries.")
f.DEFINE_string("log_gradients", True, "Whether to log gradient summaries.")
f.DEFINE_string("log_exact_match", True, "Whether to log exact match scores.")
f.DEFINE_string("log_f1_score", True, "Whether to log f1 scores.")
f.DEFINE_string("log_valid_every", 100, "Frequency (in iterations) to log " +
        "loss & gradients for the validation data set.")
f.DEFINE_integer("compute_accuracy_every", 200, "Frequency (in iterations) " +
        "to compute exact matach and f1 scores on the training and " +
        "validation data sets.")
f.DEFINE_integer("save_every", 500, "Frequency (in iterations) to save the "
        "model.")
f.DEFINE_boolean("use_s3", False,
        "Whether to use AWS S3 storage to save model checkpoints. " +
        "Checkpoints will be saved according to the experiment name and " +
        "model type.")
f.DEFINE_string("s3_bucket_name", "zippy-machine-learning",
        "The AWS S3 bucket to save models to.")
f.DEFINE_string("s3_data_folder_name", "data", "Folder within the S3 bucket " +
        "to store train/dev data and word vector indices. Only applies if " +
        "s3 storage is enabled. Using this makes it faster to start up "
        "training on another instance, rather than using SCP to upload " +
        "files and then unzip them on the EC2 instance.")
f.DEFINE_integer("num_gpus", 2, "Number of GPUs available for training. " +
        "Use 0 for CPU-only training")
f.DEFINE_integer("batch_size", 24, "Training batch size. If using GPUs, " +
        "then this will be the same for each GPU.")
f.DEFINE_integer("epochs", 10, "Number of epochs to train." +
        "The training process will keep going after this limit, but will show" +
        "a training progress percent based on this number of epochs.")
f.DEFINE_integer("num_evaluation_samples", 200, "Number of samples of the " +
        "datasets to take for partial exact match and f1 score evaluations." +
        "This is done since it can take a while to evaluate the model on the" +
        "whole dataset")
f.DEFINE_integer("rnn_size", 100, "The dimension of rnn cells.")
f.DEFINE_integer("num_rnn_layers", 1, "The number of rnn layers to use in " +
        "a single multi-rnn cell.")
f.DEFINE_float("dropout", 0.1, "The amount of dropout to use. Should be " +
        "between 0 (no dropout) and 1.0 (100% dropout).")
f.DEFINE_integer("dataset_buffer_size", 100, "Size of the dataset buffer." +
        "See the Tensorflow Dataset API for details.")
f.DEFINE_boolean("use_fake_dataset", False, "Whether to use a synthetic" +
        "dataset in order to debug the model.")
f.DEFINE_boolean("verbose_logging", False, "Whether to print verbose logs.")
f.DEFINE_float("max_global_norm", 10.0, "Used for clipping norms.")
f.DEFINE_integer("max_search_span_range", 15, "Maximum number of words in a " +
        "predicted span; used to get a small boost in performance.")
f.DEFINE_boolean("use_word_in_question_feature", True, "Whether to use the " +
        "feature indicating for each word in the context whether it is in the" +
        "question")
f.DEFINE_boolean("use_word_similarity_feature", True, "Whether to use the" +
        "feature indicating word similarity.")
f.DEFINE_boolean("use_character_data", True, "Whether to add character-level" +
        "data to the inputs.")
f.DEFINE_integer("character_embedding_size", 30, "Size of character " +
        "embeddings. Only applicable if character-level data is enabled.")
f.DEFINE_float("truncate_dataset_percent", 1.0, "Percent of the dataset" +
        "to use for training. If 1.0, then the full training set will be" +
        "used.")
f.DEFINE_integer("num_interactive_alignment_hops", 2, "Number of hops for " +
        "interactive alignment (if the model uses it).")
f.DEFINE_integer("num_memory_answer_pointer_hops", 2, "Number of hops for " +
        "the memory answer pointer model (if the model uses it).")
f.DEFINE_boolean("use_pos_tagging_feature", True, "Whether to use pos-tagging " +
        "as an extra feature to be fed into the model.")
f.DEFINE_integer("pos_embedding_size", 10, "Size of POS-tagging embedding")
f.DEFINE_boolean("use_ner_feature", True, "Whether to use named entity " +
        "recognition as an extra feature to be fed into the model.")
f.DEFINE_integer("ner_embedding_size", 10, "Size of NER embedding")
f.DEFINE_integer("sent_frag_length", 15, "Max sentence fragment length" +
        "(experimental).")
f.DEFINE_boolean("use_word_fusion_feature", True, "Whether to use the word" +
        "fusion feature as a model input.")
f.DEFINE_float("fusion_matrix_dimension", 250, "Dimension of the diagonal" +
        "matrix for fusion (FusionNet).")

def get_options_from_flags():
    return tf.app.flags.FLAGS
