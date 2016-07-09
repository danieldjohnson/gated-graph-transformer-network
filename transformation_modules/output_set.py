import theano
import theano.tensor as T
import numpy as np

from util import *
from layer import *
from graph_state import GraphState, GraphStateSpec

class OutputSetTransformation( object ):
    """
    Transforms a representation vector into an independent set output
    """
    def __init__(self, input_width, num_categories):
        self._input_width = input_width
        self._num_categories = num_categories

        self._transform_stack = LayerStack(input_width, num_categories, activation=T.nnet.sigmoid, name="output_set")

    @property
    def params(self):
        return self._transform_stack.params

    def process(self, input_vector):
        """
        Convert an input vector into a probabilistic set, i.e. a list of probabilities of item i being in
        the output set.

        Params:
            input_vector: Vector of shape (n_batch, input_width)

        Returns: Set distribution of shape (n_batch, 1, num_categories), where each value is independent from
            the others.
        """
        transformed = self._transform_stack.process(input_vector)
        return T.shape_padaxis(transformed,1)
