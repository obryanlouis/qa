import tensorflow as tf

from model.base_model import BaseModel
from model.encoding_util import *
from model.memory_answer_pointer import *
from model.qa_util import *

class QaModel(BaseModel):
    def setup(self):
        super(QaModel, self).setup()

        # Step 1. Encode the passage and question.
        ctx_dropout = tf.nn.dropout(self.ctx_inputs, self.keep_prob)
        qst_dropout = tf.nn.dropout(self.qst_inputs, self.keep_prob)
        passage_outputs, question_outputs = encode_passage_and_question(
                self.options, ctx_dropout, qst_dropout, self.keep_prob,
                self.sess, self.batch_size, self.is_train_placeholder)

        # Step 2. Run the QA loop.
        qa_ctx, qa_qst = run_qa(self.options, passage_outputs,
            question_outputs, self.keep_prob, self.is_train_placeholder,
            self.batch_size, self.sess)

        # Step 3. Get the loss & outputs with a memory answer pointer.
        self.loss, self.start_span_probs, self.end_span_probs = \
            memory_answer_pointer(self.options, qa_ctx, qa_qst,
                2 * self.options.rnn_size, self.spn_iterator, self.sq_dataset,
                self.keep_prob, self.sess, self.is_train_placeholder)
