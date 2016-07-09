import theano
import theano.tensor as T
import numpy as np

from util import *
from layer import *
from graph_state import GraphState, GraphStateSpec
from strength_weighted_gru import StrengthWeightedGRULayer

class EdgeStateUpdateTransformation( object ):
    """
    Transforms a graph state by updating edge states, conditioned on an input vector and nodes
    """
    def __init__(self, input_width, graph_spec):
        """
        Params:
            input_width: Integer giving size of input
            graph_spec: Instance of GraphStateSpec giving graph spec
        """
        self._input_width = input_width
        self._graph_spec = graph_spec
        self._process_input_size = input_width + 2*(graph_spec.num_node_ids + graph_spec.node_state_size)

        self._update_stack = LayerStack(self._process_input_size, 2*graph_spec.num_edge_types, [self._process_input_size], activation=T.nnet.sigmoid, bias_shift=-3.0, name="edge_update")
        
    @property
    def params(self):
        return self._update_stack.params

    @property
    def num_dropout_masks(self):
        return 0

    def get_dropout_masks(self, srng, keep_frac):
        return []

    def process(self, gstate, input_vector, dropout_masks=None):
        """
        Process an input vector and update the state accordingly. Each node runs a GRU step
        with previous state from the node state and input from the vector.

        Params:
            gstate: A GraphState giving the current state
            input_vector: A tensor of the form (n_batch, input_width)
        """

        # gstate.edge_states is of shape (n_batch, n_nodes, n_nodes, id+state)
        # combined input should be broadcasted to (n_batch, n_nodes, n_nodes, X)
        input_vector_part = T.shape_padaxis(T.shape_padaxis(input_vector, 1), 2)
        source_state_part = T.shape_padaxis(T.concatenate([gstate.node_ids, gstate.node_states], 2), 2)
        dest_state_part = T.shape_padaxis(T.concatenate([gstate.node_ids, gstate.node_states], 2), 1)
        full_input = broadcast_concat([input_vector_part, source_state_part, dest_state_part], 3)

        # we flatten to process updates
        flat_input = full_input.reshape([-1, self._process_input_size])
        flat_result = self._update_stack.process(flat_input)
        result = flat_result.reshape([gstate.n_batch, gstate.n_nodes, gstate.n_nodes, self._graph_spec.num_edge_types, 2])
        should_set = result[:,:,:,:,0]
        should_clear = result[:,:,:,:,1]

        new_strengths = gstate.edge_strengths*(1-should_clear) + (1-gstate.edge_strengths)*should_set

        new_gstate = gstate.with_updates(edge_strengths=new_strengths)
        return new_gstate

