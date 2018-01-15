"""Defines constants used for data preprocessing.
"""

VOCAB_CHARS_FILE = "vocab.chars.npy"
TRAIN_SQUAD_FILE = "train-v1.1.json"
DEV_SQUAD_FILE = "dev-v1.1.json"

COVE_WEIGHTS_FOLDER = "cove_weights"
COVE_WEIGHT_NAMES = ['weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0',
    'bias_hh_l0', 'weight_ih_l0_reverse', 'weight_hh_l0_reverse',
    'bias_ih_l0_reverse', 'bias_hh_l0_reverse', 'weight_ih_l1',
    'weight_hh_l1', 'bias_ih_l1', 'bias_hh_l1', 'weight_ih_l1_reverse',
    'weight_hh_l1_reverse', 'bias_ih_l1_reverse', 'bias_hh_l1_reverse']

# Training data can be split into multiple batches of files in order to
# limit the size of data in memory at once. Adjust as necessary.
MAX_SAMPLES_PER_SPLIT = 100000
TRAIN_FOLDER_NAME = "train"
DEV_FOLDER_NAME = "dev"
CONTEXT_FILE_PATTERN = "context.%d.npy"
QUESTION_FILE_PATTERN = "question.%d.npy"
SPAN_FILE_PATTERN = "span.%d.npy"
WORD_IN_QUESTION_FILE_PATTERN = "word_in_question.%d.npy"
WORD_IN_CONTEXT_FILE_PATTERN = "word_in_context.%d.npy"
QUESTION_IDS_FILE_PATTERN = "question_ids.%d.npy"
QUESTION_IDS_TO_GND_TRUTHS_FILE_PATTERN = "question_ids_to_gnd_truths.%d"
CONTEXT_POS_FILE_PATTERN = "context.pos.%d.npy"
CONTEXT_NER_FILE_PATTERN = "context.ner.%d.npy"
QUESTION_POS_FILE_PATTERN = "question.pos.%d.npy"
QUESTION_NER_FILE_PATTERN = "question.ner.%d.npy"
QUESTION_IDS_TO_SQUAD_QUESTION_ID_FILE_PATTERN = "question_ids_to_squad_question_id.%d"
QUESTION_IDS_TO_PASSAGE_CONTEXT_FILE_PATTERN = "passage_context.%d"

VECTORS_URL = "http://nlp.stanford.edu/data/glove.840B.300d.zip"
WORD_VEC_DIM = 300
MAX_WORD_LEN = 25
VECTOR_FILE = "glove.840B.300d.txt"
VECTOR_ZIP_FILE = "glove.840B.300d.zip"
SQUAD_TRAIN_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
SQUAD_TRAIN_FILE = "train-v1.1.json"
SQUAD_DEV_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
SQUAD_DEV_FILE = "dev-v1.1.json"

EMBEDDING_FILE = "glove.embedding.npy"
VOCAB_FILE = "vocab.txt"
