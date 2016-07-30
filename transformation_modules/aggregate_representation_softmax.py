import theano
import theano.tensor as T
import numpy as np

from util import *
from layer import *
from graph_state import GraphState, GraphStateSpec

class AggregateRepresentationTransformationSoftmax( object ):
    """
    Old version of the AggregateRepresentationTransformation
    Transforms a graph state into a single representation vector
    """
    def __init__(self, representation_width, graph_spec):
        self._representation_width = representation_width
        self._graph_spec = graph_spec

        self._representation_stack = LayerStack(graph_spec.num_node_ids + graph_spec.node_state_size, representation_width+1, name="aggregaterepr")
        
    @property
    def params(self):
        return self._representation_stack.params

    def process(self, gstate):
        """
        Convert the graph state to a representation vector, using softmax attention to scale representations

        Params:
            gstate: A GraphState giving the current state

        Returns: A representation vector of shape (n_batch, representation_width)
        """

        flat_obs = T.concatenate([
                        gstate.node_ids.reshape([-1, self._graph_spec.num_node_ids]),
                        gstate.node_states.reshape([-1, self._graph_spec.node_state_size])], 1)
        flat_activations = self._representation_stack.process(flat_obs)
        activations = flat_activations.reshape([gstate.n_batch, gstate.n_nodes, self._representation_width+1])

        activation_strengths = activations[:,:,0]
        existence_penalty = T.log(gstate.node_strengths + EPSILON) # TODO: consider removing epsilon here
        selector = T.shape_padright(T.nnet.softmax(activation_strengths + existence_penalty))
        representations = T.tanh(activations[:,:,1:])

        result = T.sum(selector * representations, 1)
        return result




