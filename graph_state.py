import theano
import theano.tensor as T
import numpy as np

from collections import namedtuple

GraphStateSpec = namedtuple("GraphStateSpec", ["node_state_size", "edge_state_size"])

class GraphState( object ):
    """
    A class representing the state of a graph. Wrapper for a few theano tensors
    """
    def __init__(node_strengths, node_states, edge_strengths, edge_states):
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
        Create an empty graph state with the specified sizes

            batch_size: Number of batches
            node_state_size: An integer giving size of node state
            edge_state_size: An integer givins size of edge state
        """
        return cls( T.zeros([batch_size, 0]),
                    T.zeros([batch_size, 0, node_state_size]),
                    T.zeros([batch_size, 0, 0]),
                    T.zeros([batch_size, 0, 0, edge_state_size]))

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
        next_n_nodes = next_node_strengths.shape[0]

        next_edge_strengths = T.zeros([self.n_batch, next_n_nodes, next_n_nodes])
        next_edge_strengths = T.set_subtensor(
                                    next_edge_strengths[:,:self.n_nodes,:self.n_nodes],
                                    self.edge_strengths)
        next_edge_states = T.zeros([self.n_batch, next_n_nodes, next_n_nodes, self.edge_state_width])
        next_edge_states = T.set_subtensor(
                                    next_edge_states[:,:self.n_nodes,:self.n_nodes,:],
                                    self.edge_states)
        cls = type(self)
        return cls(next_node_strengths, next_node_states, next_edge_strengths, next_edge_states)


