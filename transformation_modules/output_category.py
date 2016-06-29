import theano
import theano.tensor as T
import numpy as np

from util import *
from graph_state import GraphState, GraphStateSpec

class OutputCategoryTransformation( object ):
    """
    Transforms a representation vector into a single categorical output
    """
    def __init__(self, input_width, num_categories):
        self._input_width = input_width
        self._num_categories = num_categories

        self._transform_W = theano.shared(init_params([input_width, num_categories]), "output_category_W")
        self._transform_b = theano.shared(init_params([num_categories]), "output_category_b")

    @property
    def params(self):
        return [self._transform_W, self._transform_b]

    def process(self, input_vector):
        """
        Convert an input vector into a categorical distribution across num_categories categories

        Params:
            input_vector: Vector of shape (n_batch, input_width)

        Returns: Categorical distribution of shape (n_batch, num_categories), such that it sums to 1 across
            all categories for each instance in the batch
        """
        transformed = do_layer(T.nnet.softmax, input_vector, self._transform_W, self._transform_b)
        return transformed
