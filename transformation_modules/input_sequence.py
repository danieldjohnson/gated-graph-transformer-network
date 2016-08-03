import theano
import theano.tensor as T
import numpy as np

from util import *
from base_gru import BaseGRULayer


class InputSequenceTransformation( object ):
    """
    Transforms an input sequence into a representation vector
    """
    def __init__(self, num_words, output_width):
        self._num_words = num_words
        self._output_width = output_width

        self._gru = BaseGRULayer(num_words, output_width, name="input_sequence")

    @property
    def params(self):
        return self._gru.params

    @property
    def num_dropout_masks(self):
        return self._gru.num_dropout_masks

    def get_dropout_masks(self, srng, keep_frac):
        return self._gru.get_dropout_masks(srng, keep_frac)

    def process(self, inputs):
        """
        Process a set of inputs and return the final state

        Params:
            input_words: List of input indices. Should be an int tensor of shape (n_batch, input_len)

        Returns: The final representation vector
        """
        n_batch, input_len = inputs.shape
        valseq = T.extra_ops.to_one_hot(inputs.flatten(), self._num_words)\
                    .reshape([n_batch, input_len, self._num_words]).dimshuffle([1,0,2])
        outputs_info = [self._gru.initial_state(n_batch)]
        all_out, _ = theano.scan(self._gru.step, sequences=[valseq], outputs_info=outputs_info)

        # all_out is of shape (input_len, n_batch, self.output_width). We want last timestep
        return all_out[-1,:,:]