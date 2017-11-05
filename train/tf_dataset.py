"""Code for working with the TensorFlow Dataset api.
"""

import tensorflow as tf

CTX_KEY = "ctx"
QST_KEY = "qst"
CTX_CHAR_KEY = "ctx_char"
QST_CHAR_KEY = "qst_char"
SPN_KEY = "spn"
WIQ_KEY = "wiq"
WIC_KEY = "wic"
DATA_INDEX_KEY = "data_index"
CTX_POS_KEY = "ctx_pos"
QST_POS_KEY = "qst_pos"
CTX_NER_KEY = "ctx_ner"
QST_NER_KEY = "qst_ner"

class TfIteratorWrapper():
    def __init__(self, ctx_iterator, qst_iterator, ctx_char_iterator,
            qst_char_iterator, spn_iterator,
            data_index_iterator, word_in_question_iterator,
            word_in_context_iterator, ctx_pos_iterator, qst_pos_iterator,
            ctx_ner_iterator, qst_ner_iterator):
        print("Creating TensorFlow Datasets and iterators")
        self.ctx = ctx_iterator
        self.qst = qst_iterator
        self.ctx_chars = ctx_char_iterator
        self.qst_chars = qst_char_iterator
        self.spn = spn_iterator
        self.data_index = data_index_iterator
        self.wiq = word_in_question_iterator
        self.wic = word_in_context_iterator
        self.ctx_pos = ctx_pos_iterator
        self.qst_pos = qst_pos_iterator
        self.ctx_ner = ctx_ner_iterator
        self.qst_ner = qst_ner_iterator

class TfDataset():
    def __init__(self, options, squad_dataset):
        self.options = options
        self.train_placeholder_feed_dict = {}
        self.dev_placeholder_feed_dict = {}

        self.train_zip_ds, self.train_zip_iterator = \
            self._make_zip_ds_and_iterator(squad_dataset.train_ds,
                self.train_placeholder_feed_dict)
        self.dev_zip_ds, self.dev_zip_iterator = \
            self._make_zip_ds_and_iterator(squad_dataset.dev_ds,
                self.dev_placeholder_feed_dict)
        self.train_handle = None
        self.dev_handle = None

        self.handle = tf.placeholder(tf.string, shape=[])
        self.iterator = tf.contrib.data.Iterator.from_string_handle(
            self.handle, self.train_zip_ds.output_types, self.train_zip_ds.output_shapes)

    def setup_with_tf_session(self, sess):
        print("Setting up tensorflow data iterator handles")
        self.train_handle = sess.run(self.train_zip_iterator.string_handle())
        self.dev_handle = sess.run(self.dev_zip_iterator.string_handle())
        sess.run(self.train_zip_iterator.initializer, feed_dict=self.train_placeholder_feed_dict)
        sess.run(self.dev_zip_iterator.initializer, feed_dict=self.dev_placeholder_feed_dict)

    def get_iterator_handle(self):
        return self.handle

    def get_train_handle(self):
        return self.train_handle

    def get_dev_handle(self):
        return self.dev_handle

    def _make_zip_ds_and_iterator(self, np_dataset, placeholder_dict):
        zip_ds = tf.contrib.data.Dataset.zip({
            DATA_INDEX_KEY: self._make_ds(np_dataset.data_index, placeholder_dict),
            CTX_KEY: self._make_ds(np_dataset.ctx, placeholder_dict),
            QST_KEY: self._make_ds(np_dataset.qst, placeholder_dict),
            CTX_CHAR_KEY: self._make_ds(np_dataset.ctx_chars, placeholder_dict),
            QST_CHAR_KEY: self._make_ds(np_dataset.qst_chars, placeholder_dict),
            SPN_KEY: self._make_ds(np_dataset.spn, placeholder_dict),
            WIQ_KEY: self._make_ds(np_dataset.word_in_question, placeholder_dict),
            WIC_KEY: self._make_ds(np_dataset.word_in_context, placeholder_dict),
            CTX_POS_KEY: self._make_ds(np_dataset.context_pos, placeholder_dict),
            QST_POS_KEY: self._make_ds(np_dataset.question_pos, placeholder_dict),
            CTX_NER_KEY: self._make_ds(np_dataset.context_ner, placeholder_dict),
            QST_NER_KEY: self._make_ds(np_dataset.question_ner, placeholder_dict),
            }) \
            .shuffle(buffer_size=self.options.dataset_buffer_size)
        iterator = zip_ds.make_initializable_iterator()
        return zip_ds, iterator

    def _make_ds(self, np_arr, placeholder_dict):
        pad_shape = np_arr.shape[1:]
        placeholder = tf.placeholder(dtype=np_arr.dtype, shape=np_arr.shape)
        placeholder_dict[placeholder] = np_arr
        return tf.contrib.data.Dataset.from_tensor_slices(placeholder) \
            .padded_batch(self.options.batch_size, pad_shape) \
            .repeat()

    def create_tf_iterators(self):
        with tf.device("/cpu:0"):
            next_elem = self.iterator.get_next()
            return TfIteratorWrapper(next_elem[CTX_KEY], next_elem[QST_KEY],
                next_elem[CTX_CHAR_KEY], next_elem[QST_CHAR_KEY],
                next_elem[SPN_KEY], next_elem[DATA_INDEX_KEY],
                next_elem[WIQ_KEY], next_elem[WIC_KEY],
                next_elem[CTX_POS_KEY], next_elem[QST_POS_KEY],
                next_elem[CTX_NER_KEY], next_elem[QST_NER_KEY])
