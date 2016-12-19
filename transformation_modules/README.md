This directory contains modules implementing a set of different operations used by the model.

First, the modules that implement the five classes of graph transformation are:

- Node addition: implemented in [new_nodes_inform.py](new_nodes_inform.py)
- Node state update: non-direct-reference update implemented in [node_state_update.py](node_state_update.py), and direct-reference update implemented in [direct_reference_update.py](direct_reference_update.py)
- Edge update: implemented in [edge_state_update.py](edge_state_update.py)
- Propagation: implemented in [propagation.py](propagation.py)
- Aggregation: implemented in [aggregate_representation.py](aggregate_representation.py)

Additional modules used by the model are:

- [input_sequence_direct.py](input_sequence_direct.py) implements a GRU layer that scans through an input and creates the full sequence representation vector and the direct-reference input matrix (described in the paper in section 4).
- [output_category.py](output_category.py) implements a simple output layer that chooses a single output out of a set of possibilities. This is used for most of the bAbI tasks.
- [output_set.py](output_set.py) implements a simple output layer that chooses a single output out of a set of possibilities. This is used for task 8.
- [output_sequence.py](output_sequence.py) implements a GRU-based output layer that chooses a sequence of outputs. This is used for task 19.
- [sequence_aggregate_summary.py](sequence_aggregate_summary.py) implements a GRU-based layer that aggregates information from a series of vectors, used in implementing the version of the model described in Appendix D.

Finally, for compatibility with preliminary versions of the model, these (deprecated) modules are included:

- [aggregate_representation_softmax.py](aggregate_representation_softmax.py) implements a version of the aggregation transformation that uses softmax to select attention targets. This was found to work equivalently well to to using a sigmoid activation function for selecting attention, but was not used for final experiments for compatibility with the GG-NN model.
- [new_nodes_vote.py](new_nodes_vote.py) implements a version of the node addition transformation that uses votes from existing nodes to determine the existence of new ones, where any existing node can "veto" a new node. This was found not to work very well in initial experiments.
