import theano
import theano.tensor as T
import numpy as np

from util import *
from layer import *
from graph_state import GraphState, GraphStateSpec
from base_gru import BaseGRULayer

class PropagationTransformation( object ):
    """
    Transforms a graph state by propagating info across the graph
    """
    def __init__(self, transfer_size, graph_spec, transfer_activation=(lambda x:x), dropout_keep=1):
        """
        Params:
            transfer_size: Integer, how much to transfer
            graph_spec: Instance of GraphStateSpec giving graph spec
            transfer_activation: Activation function to use during transfer
        """
        self._transfer_size = transfer_size
        self._transfer_activation = transfer_activation
        self._graph_spec = graph_spec
        self._process_input_size = graph_spec.num_node_ids + graph_spec.node_state_size

        self._transfer_stack = LayerStack(self._process_input_size, 2 * graph_spec.num_edge_types * transfer_size, activation=self._transfer_activation, name="propagation_transfer", dropout_keep=dropout_keep, dropout_input=False, dropout_output=True)
        self._propagation_gru = BaseGRULayer(graph_spec.num_node_ids + self._transfer_size, graph_spec.node_state_size, name="propagation", dropout_keep=dropout_keep, dropout_input=False, dropout_output=True)

    @property
    def params(self):
        return self._propagation_gru.params +  self._transfer_stack.params

    def dropout_masks(self, srng):
        return self._transfer_stack.dropout_masks(srng) + self._propagation_gru.dropout_masks(srng)

    def split_dropout_masks(self, dropout_masks):
        transfer_used, dropout_masks = self._transfer_stack.split_dropout_masks(dropout_masks)
        gru_used, dropout_masks = self._propagation_gru.split_dropout_masks(dropout_masks)
        return (transfer_used+gru_used), dropout_masks

    def process(self, gstate, dropout_masks=Ellipsis):
        """
        Process a graph state.
          1. Data is transfered from each node to each other node along both forward and backward edges.
                This data is processed with a Wx+b style update, and an optional transformation is applied
          2. Nodes sum the transfered data, weighted by the existence of the other node and the edge.
          3. Nodes perform a GRU update with this input

        Params:
            gstate: A GraphState giving the current state
        """
        if dropout_masks is Ellipsis:
            dropout_masks = None
            append_masks = False
        else:
            append_masks = True

        node_obs = T.concatenate([gstate.node_ids, gstate.node_states],2)
        flat_node_obs = node_obs.reshape([-1, self._process_input_size])
        transformed, dropout_masks = self._transfer_stack.process(flat_node_obs,dropout_masks)
        transformed = transformed.reshape([gstate.n_batch, gstate.n_nodes, 2*self._graph_spec.num_edge_types, self._transfer_size])
        scaled_transformed = transformed * T.shape_padright(T.shape_padright(gstate.node_strengths))
        # scaled_transformed is of shape (n_batch, n_nodes, 2*num_edge_types, transfer_size)
        # We want to multiply  through by edge strengths, which are of shape
        # (n_batch, n_nodes, n_nodes, num_edge_types), both fwd and backward
        edge_strength_scale = T.concatenate([gstate.edge_strengths, gstate.edge_strengths.swapaxes(1,2)], 3)
        # edge_strength_scale is of (n_batch, n_nodes, n_nodes, 2*num_edge_types)
        intermed = T.shape_padaxis(scaled_transformed, 2) * T.shape_padright(edge_strength_scale)
        # intermed is of shape (n_batch, n_nodes "source", n_nodes "dest", 2*num_edge_types, transfer_size)
        # now reduce along the "source" and "edge_types" dimensions to get dest activations
        # of shape (n_batch, n_nodes, transfer_size)
        reduced_result = T.sum(T.sum(intermed, 3), 1)

        # now add information fom current node id
        full_input = T.concatenate([gstate.node_ids, reduced_result], 2)

        # we flatten to apply GRU
        flat_input = full_input.reshape([-1, self._graph_spec.num_node_ids + self._transfer_size])
        flat_state = gstate.node_states.reshape([-1, self._graph_spec.node_state_size])
        new_flat_state, dropout_masks = self._propagation_gru.step(flat_input, flat_state, dropout_masks)

        new_node_states = new_flat_state.reshape(gstate.node_states.shape)

        new_gstate = gstate.with_updates(node_states=new_node_states)
        if append_masks:
            return new_gstate, dropout_masks
        else:
            return new_gstate

    def process_multiple(self, gstate, iterations, dropout_masks=Ellipsis):
        """
        Run multiple propagagtion steps.

        Params:
            gstate: A GraphState giving the current state
            iterations: An integer. How many steps to propagate
        """
        if dropout_masks is Ellipsis:
            dropout_masks = None
            append_masks = False
        else:
            append_masks = True

        def _scan_step(cur_node_states, node_strengths, node_ids, edge_strengths, *dmasks):
            curstate = GraphState(node_strengths, node_ids, cur_node_states, edge_strengths)
            newstate, _ = self.process(curstate, dmasks if dropout_masks is not None else None)
            return newstate.node_states

        outputs_info = [gstate.node_states]
        used_dropout_masks, dropout_masks = self.split_dropout_masks(dropout_masks)
        all_node_states, _ = theano.scan(_scan_step, n_steps=iterations, non_sequences=[gstate.node_strengths, gstate.node_ids, gstate.edge_strengths] + used_dropout_masks, outputs_info=outputs_info)

        final_gstate = gstate.with_updates(node_states=all_node_states[-1,:,:,:])
        if append_masks:
            return final_gstate, dropout_masks
        else:
            return final_gstate

