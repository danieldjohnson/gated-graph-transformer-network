import theano
import theano.tensor as T
import numpy as np

from util import *
from graph_state import GraphState, GraphStateSpec
from base_gru import BaseGRULayer

class NodeStateUpdateTransformation( object ):
    """
    Transforms a graph state by updating note states, conditioned on an input vector
    """
    def __init__(self, input_width, graph_spec, dropout_keep=1):
        """
        Params:
            input_width: Integer giving size of input
            graph_spec: Instance of GraphStateSpec giving graph spec
        """
        self._input_width = input_width
        self._graph_spec = graph_spec

        self._update_gru = BaseGRULayer(input_width + graph_spec.num_node_ids, graph_spec.node_state_size, name="nodestateupdate", dropout_keep=dropout_keep, dropout_input=False, dropout_output=True)

    @property
    def params(self):
        return self._update_gru.params

    def dropout_masks(self, srng, state_mask=None):
        return self._update_gru.dropout_masks(srng, use_output=state_mask)

    def process(self, gstate, input_vector, dropout_masks=Ellipsis):
        """
        Process an input vector and update the state accordingly. Each node runs a GRU step
        with previous state from the node state and input from the vector.

        Params:
            gstate: A GraphState giving the current state
            input_vector: A tensor of the form (n_batch, input_width)
        """

        # gstate.node_states is of shape (n_batch, n_nodes, node_state_width)
        # input_vector should be broadcasted to match this
        if dropout_masks is Ellipsis:
            dropout_masks = None
            append_masks = False
        else:
            append_masks = True
        prepped_input_vector = T.tile(T.shape_padaxis(input_vector, 1), [1, gstate.n_nodes, 1])
        full_input = T.concatenate([gstate.node_ids, prepped_input_vector], 2)

        # we flatten to apply GRU
        flat_input = full_input.reshape([-1, self._input_width + self._graph_spec.num_node_ids])
        flat_state = gstate.node_states.reshape([-1, self._graph_spec.node_state_size])
        new_flat_state, dropout_masks = self._update_gru.step(flat_input, flat_state, dropout_masks)

        new_node_states = new_flat_state.reshape(gstate.node_states.shape)

        new_gstate = gstate.with_updates(node_states=new_node_states)
        if append_masks:
            return new_gstate, dropout_masks
        else:
            return new_gstate



