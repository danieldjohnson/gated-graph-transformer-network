import theano
import theano.tensor as T
import numpy as np

from util import *
from graph_state import GraphState, GraphStateSpec

class AggregateRepresentationTransformation( object ):
    """
    Transforms a graph state into a single representation vector
    """
    def __init__(self, representation_width, graph_spec):
        self._representation_width = representation_width
        self._graph_spec = graph_spec

        self._representation_W = theano.shared(init_params([graph_spec.node_state_size, representation_width+1]), "aggregaterepr_W")
        self._representation_b = theano.shared(init_params([representation_width+1]), "aggregaterepr_b")

    @property
    def params(self):
        return [self._representation_W, self._representation_b]

    def process(self, gstate):
        """
        Convert the graph state to an representation vector, using softmax attention to scale representations

        Params:
            gstate: A GraphState giving the current state

        Returns: A representation vector of shape (n_batch, representation_width)
        """

        flat_states = gstate.node_states.reshape([-1, self._graph_spec.node_state_size])
        flat_activations = do_layer(lambda x:x, flat_states, self._representation_W, self._representation_b)
        activations = flat_activations.reshape([gstate.n_batch, gstate.n_nodes, self._representation_width+1])

        selector = T.shape_padright(T.nnet.softmax(activations[:,:,0]))
        representations = T.tanh(activations[:,:,1:])

        result = T.sum(selector * representations, 1)
        return result




