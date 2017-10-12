"""Code for working with the TensorFlow Dataset api.
"""

import tensorflow as tf

CTX_KEY = "ctx"
QST_KEY = "qst"
SPN_KEY = "spn"
WIQ_KEY = "wiq"
WIC_KEY = "wic"
DATA_INDEX_KEY = "data_index"

class TfIteratorWrapper():
    def __init__(self, ctx_iterator, qst_iterator, spn_iterator,
            data_index_iterator, word_in_question_iterator,
            word_in_context_iterator):
        print("Creating TensorFlow Datasets and iterators")
        self.ctx = ctx_iterator
        self.qst = qst_iterator
        self.spn = spn_iterator
        self.data_index = data_index_iterator
        self.wiq = word_in_question_iterator
        self.wic = word_in_context_iterator

class TfDataset():
    def __init__(self, options, squad_dataset):
        self.options = options

        self.train_zip_ds, self.train_zip_iterator = \
            self._make_zip_ds_and_iterator(squad_dataset.train_ds)
        self.dev_zip_ds, self.dev_zip_iterator = \
            self._make_zip_ds_and_iterator(squad_dataset.dev_ds)
        self.train_handle = None
        self.dev_handle = None

        self.handle = tf.placeholder(tf.string, shape=[])
        self.iterator = tf.contrib.data.Iterator.from_string_handle(
            self.handle, self.train_zip_ds.output_types, self.train_zip_ds.output_shapes)

    def setup_with_tf_session(self, sess):
        self.train_handle = sess.run(self.train_zip_iterator.string_handle())
        self.dev_handle = sess.run(self.dev_zip_iterator.string_handle())

    def get_iterator_handle(self):
        return self.handle

    def get_train_handle(self):
        return self.train_handle

    def get_dev_handle(self):
        return self.dev_handle

    def _make_zip_ds_and_iterator(self, np_dataset):
        zip_ds = tf.contrib.data.Dataset.zip({
            DATA_INDEX_KEY: self._make_ds(np_dataset.data_index),
            CTX_KEY: self._make_ds(np_dataset.ctx),
            QST_KEY: self._make_ds(np_dataset.qst),
            SPN_KEY: self._make_ds(np_dataset.spn),
            WIQ_KEY: self._make_ds(np_dataset.word_in_question),
            WIC_KEY: self._make_ds(np_dataset.word_in_context),
            }) \
            .shuffle(buffer_size=self.options.dataset_buffer_size)
        iterator = zip_ds.make_one_shot_iterator()
        return zip_ds, iterator

    def _make_ds(self, np_arr):
        pad_shape = np_arr.shape[1:]
        return tf.contrib.data.Dataset.from_tensor_slices(np_arr) \
            .padded_batch(self.options.batch_size, pad_shape) \
            .repeat()

    def create_tf_iterators(self):
        with tf.device("/cpu:0"):
            next_elem = self.iterator.get_next()
            return TfIteratorWrapper(next_elem[CTX_KEY], next_elem[QST_KEY],
                next_elem[SPN_KEY], next_elem[DATA_INDEX_KEY],
                next_elem[WIQ_KEY], next_elem[WIC_KEY])
