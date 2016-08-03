import theano
import theano.tensor as T
import numpy as np

from util import *
from base_gru import BaseGRULayer


class InputSequenceDirectTransformation( object ):
    """
    Transforms an input sequence into a representation vector
    """
    def __init__(self, num_words, num_node_ids, word_node_mapping, output_width):
        """
            num_words: Number of words in the input sequence
            word_node_mapping: Mapping of word idx to node idx for direct mapping
        """
        self._num_words = num_words
        self._num_node_ids = num_node_ids
        self._word_node_mapping = word_node_mapping
        self._output_width = output_width

        self._word_node_matrix = np.zeros([num_words, num_node_ids], np.float32)
        for word,node in word_node_mapping.items():
            self._word_node_matrix[word,node] = 1.0

        self._gru = BaseGRULayer(num_words, output_width, name="input_sequence")

    @property
    def params(self):
        return self._gru.params

    def process(self, inputs):
        """
        Process a set of inputs and return the final state

        Params:
            input_words: List of input indices. Should be an int tensor of shape (n_batch, input_len)

        Returns: repr_vect, node_vects
            repr_vect: The final representation vector, of shape (n_batch, output_width)
            node_vects: Direct-access vects for each node id, of shape (n_batch, num_node_ids, output_width)
        """
        n_batch, input_len = inputs.shape
        valseq = inputs.dimshuffle([1,0])
        one_hot_vals = T.extra_ops.to_one_hot(inputs.flatten(), self._num_words)\
                    .reshape([n_batch, input_len, self._num_words])
        one_hot_valseq = one_hot_vals.dimshuffle([1,0,2])

        def scan_fn(idx_ipt, onehot_ipt, last_accum, last_state):
            # last_accum stores accumulated outputs per word type
            # and is of shape (n_batch, word_idx, output_width)
            gru_state = self._gru.step(onehot_ipt, last_state)
            new_accum = T.inc_subtensor(last_accum[T.arange(n_batch), idx_ipt, :], gru_state)
            return new_accum, gru_state

        outputs_info = [T.zeros([n_batch, self._num_words, self._output_width]), self._gru.initial_state(n_batch)]
        (all_accum, all_out), _ = theano.scan(scan_fn, sequences=[valseq, one_hot_valseq], outputs_info=outputs_info)
        
        # all_out is of shape (input_len, n_batch, self.output_width). We want last timestep
        repr_vect = all_out[-1,:,:]

        final_accum = all_accum[-1,:,:,:]
        # Now we also want to extract and accumulate the outputs that directly map to each word
        # We can do this by multipying the final accum's second dimension (word_idx) through by
        # the word_node_matrix
        resh_flat_final_accum = final_accum.dimshuffle([0,2,1]).reshape([-1, self._num_words])
        resh_flat_node_mat = T.dot(resh_flat_final_accum, self._word_node_matrix)
        node_vects = resh_flat_node_mat.reshape([n_batch, self._output_width, self._num_node_ids]).dimshuffle([0,2,1])

        return repr_vect, node_vects
