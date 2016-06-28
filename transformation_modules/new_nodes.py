import theano
import theano.tensor as T
import numpy as np

from util import *
from graph_state import GraphState, GraphStateSpec
from base_gru import BaseGRULayer

class NewNodesTransformation( object ):
    """
    Transforms a graph state by adding nodes, conditioned on an input vector
    """
    def __init__(self, input_width, graph_spec):
        """
        Params:
            input_width: Integer giving size of input
            graph_spec: Instance of GraphStateSpec giving graph spec
        """
        self._input_width = input_width
        self._graph_spec = graph_spec

        proposer_shift = [0.0]*graph_spec.node_state_size + [1.0]
        self._proposer_gru = BaseGRULayer(input_width, graph_spec.node_state_size + 1, proposer_shift, name="newnodes_proposer")

        self._vote_W = theano.shared(init_params([2*graph_spec.node_state_size, 1]), "newnodes_vote_W")
        self._vote_b = theano.shared(init_params([1], shift=1.0), "newnodes_vote_b")

    @property
    def params(self):
        return self._proposer_gru.params + [self._vote_W, self._vote_b]

    def process(self, gstate, input_vector, max_candidates):
        """
        Process an input vector and update the state accordingly. This is accomplished as follows:
          1. The proposer network, conditioned on the input vector, proposes multiple candidate nodes,
                along with a confidence
          2. Every existing node, conditioned on its own state and the candidate, votes on whether or not
                to accept this node
          3. A new node is created for each candidate node, with an existence strength given by
                confidence * [product of all votes], and an initial state state as proposed
          4. Edges to the new node are all nonexistent.

        Params:
            gstate: A GraphState giving the current state
            input_vector: A tensor of the form (n_batch, input_width)
            max_candidates: Integer, limit on the number of candidates to produce
        """
        n_batch = gstate.n_batch
        n_nodes = gstate.n_nodes
        outputs_info = [self._proposer_gru.initial_state(n_batch)]
        proposer_step = lambda st,ipt: self._proposer_gru.step(ipt,st)
        candidates, _ = theano.scan(proposer_step, n_steps=max_candidates, non_sequences=[input_vector], outputs_info=outputs_info)

        # Candidates is of shape (candidate, n_batch, node_state_size + 1)
        # Our candidate states are just the first parts (candidate, n_batch, node_state_size)
        candidate_states = candidates[:,:,:-1]
        candidate_strengths = (1 + candidates[:,:,-1])/2 #(candidate, n_batch). Note rescaling of tanh [-1,1] to be in [0,1]

        # Votes will be of shape (candidate, n_batch, n_nodes)
        # To generate this we want to assemble (candidate, n_batch, n_nodes, input_stuff),
        # squash to (parallel, input_stuff), do voting op, then unsquash
        candidate_state_part = T.shape_padaxis(candidate_states, 2)
        node_state_part = T.shape_padaxis(gstate.node_states, 0)
        full_vote_input = T.concatenate([node_state_part, candidate_state_part], 3)
        flat_vote_input = full_vote_input.reshape([-1, 2*self._graph_spec.node_state_size])
        vote_result = do_layer(T.nnet.sigmoid, flat_vote_input, self._vote_W, self._vote_b)
        final_votes = vote_result.reshape([max_candidates, n_batch, n_nodes])

        # Take the product -> (candidate, n_batch)
        chosen_strengths = candidate_strengths * T.prod(final_votes, 2)

        new_strengths = chosen_strengths.dimshuffle([1,0])
        new_states = candidate_states.dimshuffle([1,0,2])
        new_gstate = gstate.with_additional_nodes(new_strengths, new_states)

        return new_gstate


    


