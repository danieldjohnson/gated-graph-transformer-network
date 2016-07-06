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
    def __init__(self, input_width, proposal_width, graph_spec):
        """
        Params:
            input_width: Integer giving size of input
            proposal_width: Size of internal proposal
            graph_spec: Instance of GraphStateSpec giving graph spec
        """
        self._input_width = input_width
        self._graph_spec = graph_spec
        self._proposal_width = proposal_width

        self._proposer_gru = BaseGRULayer(input_width, proposal_width, name="newnodes_proposer")
        self._propose_W = theano.shared(init_params([proposal_width, 1+graph_spec.num_node_ids]), "newnodes_propose_W")
        # Bias toward making nodes. Note that we can shift everything, since softmax doesn't care about constants
        self._propose_b = theano.shared(init_params([1+graph_spec.num_node_ids], shift=1.0), "newnodes_propose_b")

        self._vote_W = theano.shared(init_params([2*graph_spec.node_id_size + graph_spec.node_state_size, 1]), "newnodes_vote_W")
        self._vote_b = theano.shared(init_params([1], shift=-1.0), "newnodes_vote_b")

    @property
    def params(self):
        return self._proposer_gru.params + [self._propose_W, self._propose_b, self._vote_W, self._vote_b]

    @property
    def num_dropout_masks(self):
        return self._proposer_gru.num_dropout_masks

    def get_dropout_masks(self, srng, keep_frac):
        return self._proposer_gru.get_dropout_masks(srng, keep_frac)

    def process(self, gstate, input_vector, max_candidates, dropout_masks=None):
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
        proposer_step = lambda st,ipt,*dm: self._proposer_gru.step(ipt,st,dm if dropout_masks is not None else None)
        raw_proposal_acts, _ = theano.scan(proposer_step, n_steps=max_candidates, non_sequences=[input_vector]+(dropout_masks if dropout_masks is not None else []), outputs_info=outputs_info)

        # raw_proposal_acts is of shape (candidate, n_batch, blah)
        flat_raw_acts = raw_proposal_acts.reshape([-1, self._proposal_width])
        flat_processed_acts = do_layer(lambda x:x, flat_raw_acts, self._propose_W, self._propose_b)
        candidate_strengths = T.nnet.sigmoid(flat_processed_acts[:,0]).reshape([max_candidates, n_batch])
        candidate_ids = T.nnet.softmax(flat_processed_acts[:,1:]).reshape([max_candidates, n_batch, self._graph_spec.num_node_ids])

        # Votes will be of shape (candidate, n_batch, n_nodes)
        # To generate this we want to assemble (candidate, n_batch, n_nodes, input_stuff),
        # squash to (parallel, input_stuff), do voting op, then unsquash
        candidate_id_part = T.shape_padaxis(candidate_ids, 2)
        node_id_part = T.shape_padaxis(gstate.node_ids, 0)
        node_state_part = T.shape_padaxis(gstate.node_states, 0)
        full_vote_input = broadcast_concat([node_id_part, node_state_part, candidate_id_part], 3)
        flat_vote_input = full_vote_input.reshape([-1, full_vote_input.shape[-1]])
        vote_result = do_layer(T.nnet.sigmoid, flat_vote_input, self._vote_W, self._vote_b)
        final_votes_no = vote_result.reshape([max_candidates, n_batch, n_nodes])
        weighted_votes_yes = 1 - final_votes_no * T.shape_padleft(gstate.node_strengths)
        # Add in the strength vote
        all_votes = T.concatenate([T.shape_padright(candidate_strengths), weighted_votes_yes], 2)
        # Take the product -> (candidate, n_batch)
        chosen_strengths = T.prod(all_votes, 2)

        new_strengths = chosen_strengths.dimshuffle([1,0])
        new_ids = candidate_ids.dimshuffle([1,0,2])
        new_gstate = gstate.with_additional_nodes(new_strengths, new_ids)

        return new_gstate


    


