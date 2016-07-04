import theano
import theano.tensor as T
import numpy as np

from util import *
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
        self._process_input_size = input_width + 2*graph_spec.node_state_size

        self._update_gru = StrengthWeightedGRULayer(self._process_input_size, graph_spec.edge_state_size, name="edgestateupdate")

    @property
    def params(self):
        return self._update_gru.params

    @property
    def num_dropout_masks(self):
        return self._update_gru.num_dropout_masks

    def get_dropout_masks(self, srng, keep_frac):
        return self._update_gru.get_dropout_masks(srng, keep_frac)

    def process(self, gstate, input_vector, dropout_masks=None):
        """
        Process an input vector and update the state accordingly. Each node runs a GRU step
        with previous state from the node state and input from the vector.

        Params:
            gstate: A GraphState giving the current state
            input_vector: A tensor of the form (n_batch, input_width)
        """

        # gstate.edge_states is of shape (n_batch, n_nodes, n_nodes, node_state_width)
        # combined input should be broadcasted to (n_batch, n_nodes, n_nodes, X)
        input_vector_part = T.shape_padaxis(T.shape_padaxis(input_vector, 1), 2)
        source_state_part = T.shape_padaxis(gstate.node_states, 2)
        dest_state_part = T.shape_padaxis(gstate.node_states, 1)
        full_input = broadcast_concat([input_vector_part, source_state_part, dest_state_part], 3)

        # we flatten to apply GRU
        flat_input = full_input.reshape([-1, self._process_input_size])
        flat_state = gstate.edge_states.reshape([-1, self._graph_spec.edge_state_size])
        flat_strength = gstate.edge_strengths.flatten()
        new_flat_state, new_flat_strength = self._update_gru.step(flat_input, flat_state, flat_strength, dropout_masks)

        new_edge_states = new_flat_state.reshape(gstate.edge_states.shape)
        new_edge_strengths = new_flat_strength.reshape(gstate.edge_strengths.shape)

        new_gstate = gstate.with_updates(edge_strengths=new_edge_strengths, edge_states=new_edge_states)
        return new_gstate

