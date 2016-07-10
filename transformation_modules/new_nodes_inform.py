import theano
import theano.tensor as T
import numpy as np

from util import *
from layer import *
from graph_state import GraphState, GraphStateSpec
from base_gru import BaseGRULayer
from .aggregate_representation import AggregateRepresentationTransformation

class NewNodesInformTransformation( object ):
    """
    Transforms a graph state by adding nodes, conditioned on an input vector
    """
    def __init__(self, input_width, inform_width, proposal_width, graph_spec):
        """
        Params:
            input_width: Integer giving size of input
            inform_width: Size of internal aggregate
            proposal_width: Size of internal proposal
            graph_spec: Instance of GraphStateSpec giving graph spec
        """
        self._input_width = input_width
        self._graph_spec = graph_spec
        self._proposal_width = proposal_width
        self._inform_width = inform_width

        self._inform_aggregate = AggregateRepresentationTransformation(inform_width, graph_spec)
        self._proposer_gru = BaseGRULayer(input_width+inform_width, proposal_width, name="newnodes_proposer")
        self._proposer_stack = LayerStack(proposal_width, 1+graph_spec.num_node_ids, [proposal_width], bias_shift=3.0, name="newnodes_proposer_post")

    @property
    def params(self):
        return self._proposer_gru.params + self._proposer_stack.params + self._inform_aggregate.params

    @property
    def num_dropout_masks(self):
        return self._proposer_gru.num_dropout_masks

    def get_dropout_masks(self, srng, keep_frac):
        return self._proposer_gru.get_dropout_masks(srng, keep_frac)

    def get_candidates(self, gstate, input_vector, max_candidates, dropout_masks=None):
        """
        Get the current candidate new nodes. This is accomplished as follows:
          1. Using the aggregate transformation, we gather information from nodes (who should have performed
                a state update already)
          1. The proposer network, conditioned on the input and info, proposes multiple candidate nodes,
                along with a confidence
          3. A new node is created for each candidate node, with an existence strength given by
                confidence, and an initial id as proposed
        This method directly returns these new nodes for comparision

        Params:
            gstate: A GraphState giving the current state
            input_vector: A tensor of the form (n_batch, input_width)
            max_candidates: Integer, limit on the number of candidates to produce

        Returns:
            new_strengths: A tensor of the form (n_batch, new_node_idx)
            new_ids: A tensor of the form (n_batch, new_node_idx, num_node_ids)
        """
        n_batch = gstate.n_batch
        n_nodes = gstate.n_nodes

        aggregated_repr = self._inform_aggregate.process(gstate)
        # aggregated_repr is of shape (n_batch, inform_width)
        
        full_input = T.concatenate([input_vector, aggregated_repr],1)

        outputs_info = [self._proposer_gru.initial_state(n_batch)]
        proposer_step = lambda st,ipt,*dm: self._proposer_gru.step(ipt,st,dm if dropout_masks is not None else None)
        raw_proposal_acts, _ = theano.scan(proposer_step, n_steps=max_candidates, non_sequences=[full_input]+(dropout_masks if dropout_masks is not None else []), outputs_info=outputs_info)

        # raw_proposal_acts is of shape (candidate, n_batch, blah)
        flat_raw_acts = raw_proposal_acts.reshape([-1, self._proposal_width])
        flat_processed_acts = self._proposer_stack.process(flat_raw_acts)
        candidate_strengths = T.nnet.sigmoid(flat_processed_acts[:,0]).reshape([max_candidates, n_batch])
        candidate_ids = T.nnet.softmax(flat_processed_acts[:,1:]).reshape([max_candidates, n_batch, self._graph_spec.num_node_ids])

        new_strengths = candidate_strengths.dimshuffle([1,0])
        new_ids = candidate_ids.dimshuffle([1,0,2])
        return new_strengths, new_ids

    def process(self, gstate, input_vector, max_candidates, dropout_masks=None):
        """
        Process an input vector and update the state accordingly.
        """
        new_strengths, new_ids = self.get_candidates(gstate, input_vector, max_candidates, dropout_masks)
        new_gstate = gstate.with_additional_nodes(new_strengths, new_ids)
        return new_gstate


    


