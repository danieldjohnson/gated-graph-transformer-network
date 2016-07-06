import theano
import theano.tensor as T
import numpy as np
from util import *

from collections import namedtuple

GraphStateSpec = namedtuple("GraphStateSpec", ["num_node_ids", "node_state_size", "num_edge_types"])

class GraphState( object ):
    """
    A class representing the state of a graph. Wrapper for a few theano tensors
    """
    def __init__(self, node_strengths, node_ids, node_states, edge_strengths):
        """
        Create a graph state directly from existing nodes and edges.
            
            node_strengths: Tensor of shape (batch, n_nodes)
            node_ids: Tensor of shape (batch, n_nodes, num_node_ids)
            node_states: Tensor of shape (batch, n_nodes, node_state_size)
            edge_strengths: Tensor of shape (batch, n_nodes, n_nodes, num_edge_types)
        """
        self._node_strengths = node_strengths
        self._node_ids = node_ids
        self._node_states = node_states
        self._edge_strengths = edge_strengths

    @classmethod
    def create_empty(cls, batch_size, num_node_ids, node_state_size, num_edge_types):
        """
        Create an empty graph state with the specified sizes. Note that this
        will contain one zero-strength element to prevent nasty GPU errors
        from a dimension with 0 in it.

            batch_size: Number of batches
            num_node_ids: An integer giving size of node id
            node_state_size: An integer giving size of node state
            num_edge_types: An integer giving number of edge types
        """
        return cls( T.unbroadcast(T.zeros([batch_size, 1]), 1),
                    T.unbroadcast(T.zeros([batch_size, 1, num_node_ids]), 1),
                    T.unbroadcast(T.zeros([batch_size, 1, node_state_size]), 1),
                    T.unbroadcast(T.zeros([batch_size, 1, 1, num_edge_types]), 1, 2))

    @classmethod
    def create_empty_from_spec(cls, batch_size, spec):
        """
        Create an empty graph state from a spec

            batch_size: Number of batches
            spec: Instance of GraphStateSpec
        """
        return cls.create_empty(batch_size, spec.num_node_ids, spec.node_state_size, spec.num_edge_types)

    @classmethod
    def create_full_unique(cls, batch_size, num_node_ids, node_state_size, num_edge_types):
        """
        Create a 'full unique' graph state (i.e. a graph state where every id has exactly one node) from a spec

            batch_size: Number of batches
            num_node_ids: An integer giving size of node id
            node_state_size: An integer giving size of node state
            num_edge_types: An integer giving number of edge types
        """
        return cls( T.ones([batch_size, num_node_ids]),
                    T.tile(T.shape_padleft(T.eye(num_node_ids)), (batch_size,1,1)),
                    T.zeros([batch_size, num_node_ids, node_state_size]),
                    T.zeros([batch_size, num_node_ids, num_node_ids, num_edge_types]))

    @classmethod
    def create_full_unique_from_spec(cls, batch_size, spec):
        """
        Create a 'full unique' graph state (i.e. a graph state where every id has exactly one node) from a spec

            batch_size: Number of batches
            spec: Instance of GraphStateSpec
        """
        return cls.create_full_unique(batch_size, spec.num_node_ids, spec.node_state_size, spec.num_edge_types)

    @property
    def node_strengths(self):
        return self._node_strengths

    @property
    def node_states(self):
        return self._node_states

    @property
    def node_ids(self):
        return self._node_ids

    @property
    def edge_strengths(self):
        return self._edge_strengths

    @property
    def n_batch(self):
        return self.node_states.shape[0]

    @property
    def n_nodes(self):
        return self.node_states.shape[1]

    @property
    def node_id_width(self):
        return self.node_ids.shape[2]

    @property
    def node_state_width(self):
        return self.node_states.shape[2]

    @property
    def num_edge_types(self):
        return self.edge_strengths.shape[3]
    
    def flatten(self):
        return [self.node_strengths, self.node_ids, self.node_states, self.edge_strengths]

    @classmethod
    def unflatten(cls, vals):
        return cls(*vals)

    @classmethod
    def const_flattened_length(cls):
        return 5

    def flatten_to_const_size(self, const_n_nodes):
        exp_node_strengths = pad_to(self.node_strengths, [self.n_batch, const_n_nodes])
        exp_node_ids = pad_to(self.node_ids, [self.n_batch, const_n_nodes, self.node_id_width])
        exp_node_states = pad_to(self.node_states, [self.n_batch, const_n_nodes, self.node_state_width])
        exp_edge_strengths = pad_to(self.edge_strengths, [self.n_batch, const_n_nodes, const_n_nodes, self.num_edge_types])
        return [exp_node_strengths, exp_node_ids, exp_node_states, exp_edge_strengths, self.n_nodes]
    
    @classmethod
    def unflatten_from_const_size(cls, vals):
        exp_node_strengths, exp_node_ids, exp_node_states, exp_edge_strengths, n_nodes = vals
        return cls( exp_node_strengths[:,:n_nodes],
                    exp_node_ids[:,:n_nodes,:],
                    exp_node_states[:,:n_nodes,:],
                    exp_edge_strengths[:,:n_nodes,:n_nodes,:])

    def with_updates(self, node_strengths=None, node_ids=None, node_states=None, edge_strengths=None):
        """
        Helper function to generate a new state with changes applied. Params like in constructor, or None
        to use current values

        Returns: A new graph state with the changes
        """
        node_strengths = self.node_strengths if node_strengths is None else node_strengths
        node_ids = self.node_ids if node_ids is None else node_ids
        node_states = self.node_states if node_states is None else node_states
        edge_strengths = self.edge_strengths if edge_strengths is None else edge_strengths
        cls = type(self)
        return cls(node_strengths, node_ids, node_states, edge_strengths)

    def with_additional_nodes(self, new_node_strengths, new_node_ids, new_node_states=None):
        """
        Helper function to generate a new state with new nodes added.

        Params:
            new_node_strengths: Tensor of shape (n_batch, n_new_nodes)
            new_node_ids: Tensor of shape (n_batch, n_new_nodes, num_node_ids)
            new_node_states: (Optional) Tensor of shape (n_batch, n_new_nodes, node_state_size)
                If not provided, will be zero

        Returns: A new graph state with the changes
        """
        if new_node_states is None:
            new_node_states = T.zeros([self.n_batch, new_node_strengths.shape[1], self.node_state_size])

        next_node_strengths = T.concatenate([self.node_strengths, new_node_strengths], 1)
        next_node_ids = T.concatenate([self.node_ids, new_node_states], 1)
        next_node_states = T.concatenate([self.node_states, new_node_states], 1)
        next_n_nodes = next_node_strengths.shape[1]

        next_edge_strengths = pad_to(self.edge_strengths, [self.n_batch, next_n_nodes, next_n_nodes, self.num_edge_types])

        cls = type(self)
        return cls(next_node_strengths, next_node_ids, next_node_states, next_edge_strengths)


