import theano
import theano.tensor as T
import numpy as np

from util import *
from graph_state import GraphState, GraphStateSpec
from base_gru import BaseGRULayer


class OutputSequenceTransformation( object ):
    """
    Transforms a representation vector into a sequence of outputs
    """
    def __init__(self, input_width, state_size, num_words):
        self._input_width = input_width
        self._state_size = state_size
        self._num_words = num_words

        self._seq_gru = BaseGRULayer(input_width, state_size, name="output_seq_gru")
        self._transform_W = theano.shared(init_params([state_size, num_words]), "output_seq_transf_W")
        self._transform_b = theano.shared(init_params([num_words]), "output_seq_transf_b")

    @property
    def params(self):
        return self._seq_gru.params + [self._transform_W, self._transform_b]

    def process(self, input_vector, seq_len):
        """
        Convert an input vector into a sequence of categorical distributions

        Params:
            input_vector: Vector of shape (n_batch, input_width)
            seq_len: How many outputs to produce

        Returns: Set distribution of shape (n_batch, num_categories), where each value is independent from
            the others.
        """
        n_batch = input_vector.shape[0]
        outputs_info = [self._seq_gru.initial_state(n_batch)]
        scan_step = lambda state, ipt: self._seq_gru.step(ipt, state)
        all_out, _ = theano.scan(scan_step, non_sequences=[input_vector], n_steps=seq_len, outputs_info=outputs_info)

        # all_out is of shape (seq_len, n_batch, state_size). Squash and apply layer
        flat_out = all_out.reshape([-1, self._state_size])
        flat_final = do_layer(T.nnet.softmax, flat_out, self._transform_W, self._transform_b)
        final = flat_final.reshape([seq_len, n_batch, self._num_words])

        return final
