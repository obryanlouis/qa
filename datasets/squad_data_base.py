"""Defines a base class for the SQuAD datasets (debug or real data).
"""

from abc import ABCMeta, abstractmethod

class SquadDataBase:
    def get_max_ctx_len(self):
        return self.train_ds.ctx.shape[1]

    def get_max_qst_len(self):
        return self.train_ds.qst.shape[1]

    @abstractmethod
    def get_word_vec_size(self):
        pass
