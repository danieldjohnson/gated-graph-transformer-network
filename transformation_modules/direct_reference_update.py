import theano
import theano.tensor as T
import numpy as np

from util import *
from graph_state import GraphState, GraphStateSpec
from base_gru import BaseGRULayer

class DirectReferenceUpdateTransformation( object ):
    """
    Transforms a graph state by updating note states, conditioned on a direct reference accumulation
    """
    def __init__(self, input_width, graph_spec, dropout_keep=1):
        """
        Params:
            input_width: Integer giving size of input
            graph_spec: Instance of GraphStateSpec giving graph spec
        """
        self._input_width = input_width
        self._graph_spec = graph_spec

        self._update_gru = BaseGRULayer(input_width + graph_spec.num_node_ids, graph_spec.node_state_size, name="nodestateupdate", dropout_keep=dropout_keep)

    @property
    def params(self):
        return self._update_gru.params

    def dropout_masks(self, srng):
        return self._update_gru.dropout_masks(srng)

    def process(self, gstate, ref_matrix, dropout_masks):
        """
        Process a direct ref matrix and update the state accordingly. Each node runs a GRU step
        with previous state from the node state and input from the matrix.

        Params:
            gstate: A GraphState giving the current state
            ref_matrix: A tensor of the form (n_batch, num_node_ids, input_width)
        """

        # To process the input, we need to map from node id to node index
        # We can do this using the gstate.node_ids, of shape (n_batch, n_nodes, num_node_ids)
        prepped_input_vector = T.batched_dot(gstate.node_ids, ref_matrix)

        # prepped_input_vector is of shape (n_batch, n_nodes, input_width)
        # gstate.node_states is of shape (n_batch, n_nodes, node_state_width)
        # so they match nicely
        full_input = T.concatenate([gstate.node_ids, prepped_input_vector], 2)

        # we flatten to apply GRU
        flat_input = full_input.reshape([-1, self._input_width + self._graph_spec.num_node_ids])
        flat_state = gstate.node_states.reshape([-1, self._graph_spec.node_state_size])
        new_flat_state, dropout_masks = self._update_gru.step(flat_input, flat_state, dropout_masks)

        new_node_states = new_flat_state.reshape(gstate.node_states.shape)

        new_gstate = gstate.with_updates(node_states=new_node_states)
        return new_gstate, dropout_masks



