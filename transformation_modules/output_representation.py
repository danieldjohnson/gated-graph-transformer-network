import theano
import theano.tensor as T
import numpy as np

from util import *
from graph_state import GraphState, GraphStateSpec

class OutputRepresentationTransformation( object ):
    """
    Transforms a graph state into an output representation
    """
    def __init__(self, output_width, graph_spec):
        self._output_width = output_width
        self._graph_spec = graph_spec

        self._output_W = theano.shared(init_params([graph_spec.node_state_size, output_width+1]), "output_W")
        self._output_b = theano.shared(init_params([output_width+1]), "output_b")

    @property
    def params(self):
        return [self._output_W, self._output_b]

    def process(self, gstate):
        """
        Convert the graph state to an output vector, using softmax attention to scale outputs

        Params:
            gstate: A GraphState giving the current state

        Returns: A representation vector of shape (n_batch, output_width)
        """

        flat_states = gstate.node_states.reshape([-1, self._graph_spec.node_state_size])
        flat_activations = do_layer(lambda x:x, flat_states, self._output_W, self._output_b)
        activations = flat_activations.reshape([gstate.n_batch, gstate.n_nodes, self._output_width+1])

        selector = T.shape_padright(T.nnet.softmax(activations[:,:,0]))
        representations = T.tanh(activations[:,:,1:])

        result = T.sum(selector * representations, 1)
        return result




