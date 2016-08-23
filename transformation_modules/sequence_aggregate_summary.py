import theano
import theano.tensor as T
import numpy as np

from util import *
from layer import *
from graph_state import GraphState, GraphStateSpec
from base_gru import BaseGRULayer

class SequenceAggregateSummaryTransformation( object ):
    """
    Transforms a sequence of aggregate representation vectors into a summary vector
    """
    def __init__(self, input_representation_width, output_representation_width, dropout_keep=1):
        self._input_representation_width = input_representation_width
        self._output_representation_width = output_representation_width

        self._seq_gru = BaseGRULayer(input_representation_width, output_representation_width, dropout_keep=dropout_keep, name="summary_seq_gru")

    @property
    def params(self):
        return self._seq_gru.params

    def dropout_masks(self, srng):
        return self._seq_gru.dropout_masks(srng)

    def process(self, input_sequence, dropout_masks=Ellipsis):
        """
        Convert the sequence of vectors to a summary vector
        Params:
            input_sequence: A tensor of shape (n_batch, time, input_representation_width)

        Returns: A representation vector of shape (n_batch, output_representation_width)
        """
        if dropout_masks is Ellipsis:
            dropout_masks = None
            append_masks = False
        else:
            append_masks = True

        n_batch = input_sequence.shape[0]
        swapped_input = input_sequence.swapaxes(0,1)
        outputs_info = [self._seq_gru.initial_state(n_batch)]
        scan_step = lambda ipt, state, *dmasks: self._seq_gru.step(ipt, state, None if dropout_masks is None else dmasks)[0]
        all_out, _ = theano.scan(scan_step, sequences=[swapped_input], non_sequences=dropout_masks, outputs_info=outputs_info)

        result = all_out[-1]

        if append_masks:
            return result, dropout_masks
        else:
            return result
