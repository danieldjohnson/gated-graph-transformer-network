import theano
import theano.tensor as T
import numpy as np

from util import *
from layer import *
from graph_state import GraphState, GraphStateSpec

class OutputCategoryTransformation( object ):
    """
    Transforms a representation vector into a single categorical output
    """
    def __init__(self, input_width, num_categories):
        self._input_width = input_width
        self._num_categories = num_categories

        self._transform_stack = LayerStack(input_width, num_categories, activation=T.nnet.softmax, name="output_category")

    @property
    def params(self):
        return self._transform_stack.params

    def process(self, input_vector):
        """
        Convert an input vector into a categorical distribution across num_categories categories

        Params:
            input_vector: Vector of shape (n_batch, input_width)

        Returns: Categorical distribution of shape (n_batch, 1, num_categories), such that it sums to 1 across
            all categories for each instance in the batch
        """
        transformed, _ = self._transform_stack.process(input_vector, None)
        return T.shape_padaxis(transformed,1)

    def snap_to_best(self, answer):
        """
        Convert output of process to the "best" answer, i.e. the answer with highest probability.
        """
        return categorical_best(answer)
