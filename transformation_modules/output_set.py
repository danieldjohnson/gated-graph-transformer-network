import theano
import theano.tensor as T
import numpy as np

from util import *
from graph_state import GraphState, GraphStateSpec

class OutputSetTransformation( object ):
    """
    Transforms a representation vector into an independent set output
    """
    def __init__(self, input_width, num_categories):
        self._input_width = input_width
        self._num_categories = num_categories

        self._transform_W = theano.shared(init_params([input_width, num_categories]), "output_set_W")
        self._transform_b = theano.shared(init_params([num_categories]), "output_set_b")

    @property
    def params(self):
        return [self._transform_W, self._transform_b]

    def process(self, input_vector):
        """
        Convert an input vector into a probabilistic set, i.e. a list of probabilities of item i being in
        the output set.

        Params:
            input_vector: Vector of shape (n_batch, input_width)

        Returns: Set distribution of shape (n_batch, num_categories), where each value is independent from
            the others.
        """
        transformed = do_layer(T.nnet.sigmoid, input_vector, self._transform_W, self._transform_b)
        return transformed
