"""Implements a phase conductor model.
https://arxiv.org/pdf/1710.10504.pdf
"""

import tensorflow as tf

from model.base_model import BaseModel
from model.conductor_net_encoder import *
from model.conductor_net_outer_fusion import *
from model.encoding_util import *
from model.memory_answer_pointer import *
from model.rnn_util import *

class ConductorNet(BaseModel):
    def setup(self):
        super(ConductorNet, self).setup()
        # Step 1. Encode the passage and question.
        encoded_ctx, encoded_qst = encode_conductor_net(self.ctx_inputs,
            self.qst_inputs, self.keep_prob, self.use_dropout_placeholder,
            self.batch_size, self.options, self.sess) # size = [batch_size, max_ctx_length, 2 * rnn_size * num_conductor_net_encoder_layers]
        # Step 2. Run outer fusion layers.
        outer_fusion = conductor_net_outer_fusion(self.options, encoded_ctx,
            self.keep_prob) # size = [batch_size, max_ctx_length, 2 * rnn_size * num_conductor_net_encoder_layers]
        # Step 3. Run self attention layers with inner fusion.
        self_attention = conductor_net_self_attention(self.options,
            outer_fusion, self.keep_prob)
        # Step 4. Use a memory-based answer pointer mechanism to get the loss,
        # and start & end span probabilities
        self.loss, self.start_span_probs, self.end_span_probs = \
            memory_answer_pointer(self.options, self_attention, encoded_qst,
                ctx_dim, self.spn_iterator, self.sq_dataset, self.keep_prob,
                self.sess, self.use_dropout_placeholder)
