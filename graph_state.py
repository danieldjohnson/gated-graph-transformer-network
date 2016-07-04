import theano
import theano.tensor as T
import numpy as np
from util import *

from collections import namedtuple

GraphStateSpec = namedtuple("GraphStateSpec", ["node_state_size", "edge_state_size"])

class GraphState( object ):
    """
    A class representing the state of a graph. Wrapper for a few theano tensors
    """
    def __init__(self, node_strengths, node_states, edge_strengths, edge_states):
        """
        Create a graph state directly from existing nodes and edges.
            
            node_strengths: Tensor of shape (batch, n_nodes)
            node_states: Tensor of shape (batch, n_nodes, node_state_width)
            edge_strengths: Tensor of shape (batch, n_nodes, n_nodes)
            edge_states: Tensor of shape (batch, n_nodes, n_nodes, edge_state_width)
        """
        self._node_strengths = node_strengths
        self._node_states = node_states
        self._edge_strengths = edge_strengths
        self._edge_states = edge_states

    @classmethod
    def create_empty(cls, batch_size, node_state_size, edge_state_size):
        """
        Create an empty graph state with the specified sizes. Note that this
        will contain one zero-strength element to prevent nasty GPU errors
        from a dimension with 0 in it.

            batch_size: Number of batches
            node_state_size: An integer giving size of node state
            edge_state_size: An integer givins size of edge state
        """
        return cls( T.unbroadcast(T.zeros([batch_size, 1]), 1),
                    T.unbroadcast(T.zeros([batch_size, 1, node_state_size]), 1),
                    T.unbroadcast(T.zeros([batch_size, 1, 1]), 1, 2),
                    T.unbroadcast(T.zeros([batch_size, 1, 1, edge_state_size]), 1, 2))

    @classmethod
    def create_empty_from_spec(cls, batch_size, spec):
        """
        Create an empty graph state from a spec

            batch_size: Number of batches
            spec: Instance of GraphStateSpec
        """
        return cls.create_empty(batch_size, spec.node_state_size, spec.edge_state_size)

    @property
    def node_strengths(self):
        return self._node_strengths

    @property
    def node_states(self):
        return self._node_states

    @property
    def edge_strengths(self):
        return self._edge_strengths

    @property
    def edge_states(self):
        return self._edge_states

    @property
    def n_batch(self):
        return self.node_states.shape[0]

    @property
    def n_nodes(self):
        return self.node_states.shape[1]

    @property
    def node_state_width(self):
        return self.node_states.shape[2]

    @property
    def edge_state_width(self):
        return self.edge_states.shape[3]
    
    def flatten(self):
        return [self.node_strengths, self.node_states, self.edge_strengths, self.edge_states]

    @classmethod
    def unflatten(cls, vals):
        return cls(*vals)

    @classmethod
    def const_flattened_length(cls):
        return 5

    def flatten_to_const_size(self, const_n_nodes):
        exp_node_strengths = pad_to(self.node_strengths, [self.n_batch, const_n_nodes])
        exp_node_states = pad_to(self.node_states, [self.n_batch, const_n_nodes, self.node_state_width])
        exp_edge_strengths = pad_to(self.edge_strengths, [self.n_batch, const_n_nodes, const_n_nodes])
        exp_edge_states = pad_to(self.edge_states, [self.n_batch, const_n_nodes, const_n_nodes, self.edge_state_width])
        return [exp_node_strengths, exp_node_states, exp_edge_strengths, exp_edge_states, self.n_nodes]
    
    @classmethod
    def unflatten_from_const_size(cls, vals):
        exp_node_strengths, exp_node_states, exp_edge_strengths, exp_edge_states, n_nodes = vals
        return cls( exp_node_strengths[:,:n_nodes],
                    exp_node_states[:,:n_nodes,:],
                    exp_edge_strengths[:,:n_nodes,:n_nodes],
                    exp_edge_states[:,:n_nodes,:n_nodes,:])

    def with_updates(self, node_strengths=None, node_states=None, edge_strengths=None, edge_states=None):
        """
        Helper function to generate a new state with changes applied. Params like in constructor, or None
        to use current values

        Returns: A new graph state with the changes
        """
        node_strengths = self.node_strengths if node_strengths is None else node_strengths
        node_states = self.node_states if node_states is None else node_states
        edge_strengths = self.edge_strengths if edge_strengths is None else edge_strengths
        edge_states = self.edge_states if edge_states is None else edge_states
        cls = type(self)
        return cls(node_strengths, node_states, edge_strengths, edge_states)

    def with_additional_nodes(self, new_node_strengths, new_node_states):
        """
        Helper function to generate a new state with new nodes added.

        Params:
            new_node_strengths: Tensor of shape (n_batch, n_new_nodes)
            new_node_states: Tensor of shape (n_batch, n_new_nodes, node_state_size)

        Returns: A new graph state with the changes
        """
        next_node_strengths = T.concatenate([self.node_strengths, new_node_strengths], 1)
        next_node_states = T.concatenate([self.node_states, new_node_states], 1)
        next_n_nodes = next_node_strengths.shape[1]

        next_edge_strengths = pad_to(self.edge_strengths, [self.n_batch, next_n_nodes, next_n_nodes])
        next_edge_states = pad_to(self.edge_states, [self.n_batch, next_n_nodes, next_n_nodes, self.edge_state_width])

        cls = type(self)
        return cls(next_node_strengths, next_node_states, next_edge_strengths, next_edge_states)


