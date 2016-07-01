import theano
import theano.tensor as T
import numpy as np

from util import *
from graph_state import GraphState, GraphStateSpec
from base_gru import BaseGRULayer

class PropagationTransformation( object ):
    """
    Transforms a graph state by propagating info across the graph
    """
    def __init__(self, transfer_size, graph_spec, transfer_activation=(lambda x:x)):
        """
        Params:
            transfer_size: Integer, how much to transfer
            graph_spec: Instance of GraphStateSpec giving graph spec
            transfer_activation: Activation function to use during transfer
        """
        self._transfer_size = transfer_size
        self._transfer_activation = transfer_activation
        self._graph_spec = graph_spec
        self._process_input_size = graph_spec.node_state_size + graph_spec.edge_state_size

        self._transfer_fwd_W = theano.shared(init_params([self._process_input_size, transfer_size]), "propagation_transfer_fwd_W")
        self._transfer_fwd_b = theano.shared(init_params([transfer_size]), "propagation_transfer_fwd_b")
        self._transfer_bwd_W = theano.shared(init_params([self._process_input_size, transfer_size]), "propagation_transfer_bwd_W")
        self._transfer_bwd_b = theano.shared(init_params([transfer_size]), "propagation_transfer_bwd_b")

        self._propagation_gru = BaseGRULayer(self._transfer_size, graph_spec.node_state_size, name="propagation")

    @property
    def params(self):
        return self._propagation_gru.params + [self._transfer_fwd_W, self._transfer_fwd_b, self._transfer_bwd_W, self._transfer_bwd_b]

    def process(self, gstate):
        """
        Process a graph state.
          1. Data is transfered from each node to each other node along both forward and backward edges.
                This data is processed with a Wx+b style update, and an optional transformation is applied
          2. Nodes sum the transfered data, weighted by the existence of the other node and the edge.
          3. Nodes perform a GRU update with this input

        Params:
            gstate: A GraphState giving the current state
        """

        def helper_transform(aligned_source_part, transfer_W, transfer_b):
            # combined input should be broadcasted to (n_batch, n_nodes, n_nodes, X)
            edge_state_part = gstate.edge_states
            full_input = broadcast_concat([aligned_source_part, edge_state_part], 3)
            flat_input = full_input.reshape([-1, self._process_input_size])
            transformed = do_layer(self._transfer_activation, flat_input, transfer_W, transfer_b)\
                            .reshape([gstate.n_batch, gstate.n_nodes, gstate.n_nodes, self._transfer_size])
            # now transformed has shape (n_batch, n_nodes, n_nodes, _transfer_size)
            # scale by edge strength
            edge_strength_scale = T.shape_padright(gstate.edge_strengths)
            return transformed * edge_strength_scale

        source_state_part = T.shape_padaxis(gstate.node_states, 2)
        fwd_result = helper_transform(source_state_part, self._transfer_fwd_W, self._transfer_fwd_b)

        dest_state_part = T.shape_padaxis(gstate.node_states, 1)
        bwd_result = helper_transform(dest_state_part, self._transfer_bwd_W, self._transfer_bwd_b).dimshuffle([0,2,1,3])

        combined_result = fwd_result + bwd_result
        # combined_result is of shape (n_batch, n_nodes, n_nodes, _transfer_size), where
        # index 1 is the "from" node (where the info came from), index 2 is the "to" node
        # now we scale the result by the strength the "from" nodes, and reduce across
        # "from" nodes to produce the update input for each "to" node
        node_strength_scale = T.shape_padright(T.shape_padright(gstate.node_strengths))
        reduced_result = T.sum(node_strength_scale * combined_result, 1)

        # we flatten to apply GRU
        flat_input = reduced_result.reshape([-1, self._transfer_size])
        flat_state = gstate.node_states.reshape([-1, self._graph_spec.node_state_size])
        new_flat_state = self._propagation_gru.step(flat_input, flat_state)

        new_node_states = new_flat_state.reshape(gstate.node_states.shape)

        new_gstate = gstate.with_updates(node_states=new_node_states)
        return new_gstate

    def process_multiple(self, gstate, iterations):
        """
        Run multiple propagagtion steps.

        Params:
            gstate: A GraphState giving the current state
            iterations: An integer. How many steps to propagate
        """

        def _scan_step(cur_node_states, node_strengths, edge_strengths, edge_states):
            curstate = GraphState(node_strengths, cur_node_states, edge_strengths, edge_states)
            return self.process(curstate).node_states

        outputs_info = [gstate.node_states]
        all_node_states, _ = theano.scan(_scan_step, n_steps=iterations, non_sequences=[gstate.node_strengths, gstate.edge_strengths, gstate.edge_states], outputs_info=outputs_info)

        final_gstate = gstate.with_updates(node_states=all_node_states[-1,:,:,:])
        return final_gstate

