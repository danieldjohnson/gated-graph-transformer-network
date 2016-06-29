import theano
import theano.tensor as T
import numpy as np

from enum import Enum
import itertools
import transformation_modules as tfms
from graph_state import GraphStateSpec, GraphState 
from adam import Adam

class ModelOutputFormat( Enum )
    category = 1
    subset = 2
    sequence = 3

class Model( object ):
    """
    Implements the gated graph memory network model. 
    """

    def __init__(self, num_input_words, num_output_words, node_state_size, edge_state_size, input_repr_size, output_repr_size, propagate_repr_size, new_nodes_per_iter, output_format,final_propagate, node_state_update=True, intermediate_propagate=0, setup=True):
        """
        Parameters:
            num_input_words: How many possible words in the input
            num_output_words: How many possible words in the output
            node_state_size: State size for nodes
            edge_state_size: State size for edges
            input_repr_size: Width of the intermediate input representation given to the network
            output_repr_size: Width of the intermediate output representation produced by the network
            propagate_repr_size: Width of the intermediate propagation representation
            new_nodes_per_iter: How many nodes to add at each sentence iteration
            output_format: Member of ModelOutputFormat, giving the format of the output
            final_propagate: How many steps to propagate info for each input sentence
            intermediate_propagate: How many steps to propagate info for each input sentence
            node_state_update: Whether nodes should update their state based on input
            setup: Whether or not to automatically set up the model
        """
        self.num_input_words = num_input_words
        self.num_output_words = num_output_words
        self.node_state_size = node_state_size
        self.edge_state_size = edge_state_size
        self.input_repr_size = input_repr_size
        self.output_repr_size = output_repr_size
        self.propagate_repr_size = propagate_repr_size
        self.new_nodes_per_iter = new_nodes_per_iter
        self.output_format = output_format
        self.final_propagate = final_propagate
        self.intermediate_propagate = intermediate_propagate
        self.node_state_update = node_state_update

        graphspec = GraphStateSpec(self.node_state_size, self.edge_state_size)

        self.parameterized = []

        self.input_transformer = tfms.InputSequenceTransformation(num_input_words, input_repr_size)
        self.parameterized.append(self.input_transformer)

        if node_state_update:
            self.node_state_updater = tfms.NodeStateUpdateTransformation(input_repr_size, graphspec)
            self.parameterized.append(self.node_state_updater)

        if intermediate_propagate != 0:
            self.intermediate_propagator = tfms.PropagationTransformation(propagate_repr_size, graphspec, T.tanh)
            self.parameterized.append(self.intermediate_propagator)

        self.new_node_adder = tfms.NewNodesTransformation(input_repr_size, graphspec)
        self.parameterized.append(self.new_node_adder)

        self.edge_state_updater = tfms.EdgeStateUpdateTransformation(input_repr_size, graphspec)
        self.parameterized.append(self.edge_state_updater)

        self.final_propagator = tfms.PropagationTransformation(propagate_repr_size, graphspec, T.tanh)
        self.parameterized.append(self.final_propagator)

        self.aggregator = tfms.AggregateRepresentationTransformation(output_repr_size, graphspec)
        self.parameterized.append(self.aggregator)

        assert output_format in ModelOutputFormat, "Invalid output format {}".format(output_format)
        if output_format == ModelOutputFormat.category:
            self.output_processor = tfms.OutputCategoryTransformation(output_repr_size, num_output_words)
        elif output_format == ModelOutputFormat.subset:
            self.output_processor = tfms.OutputSetTransformation(output_repr_size, num_output_words)
        elif output_format == ModelOutputFormat.sequence:
            self.output_processor = tfms.OutputSequenceTransformation(output_repr_size, output_repr_size, num_output_words)
        self.parameterized.append(self.output_processor)

        if setup:
            self.setup()

    @property
    def params(self):
        return list(itertools.chain(*(l.params for l in self.parameterized)))

    def setup(self):
        """
        Set up the model to train.
        """

        # input_words: shape (n_batch, n_sentence, sentence_len)
        input_words = T.itensor3()
        n_batch, n_sentences, sentence_len = input_words.shape

        # Process each sentence, flattened to (?, sentence_len)
        flat_input_words = input_words.reshape([-1, sentence_len])
        flat_input_reprs = self.input_transformer.process(flat_input_words) # shape (?, input_repr_size)
        input_reprs = flat_input_reprs.reshape([n_batch, n_sentences, self.input_repr_size])

        # Scan over each sentence
        def _scan_fn(input_repr, *stuff) # (input_repr, *flat_graph_state, pad_graph_size)
            pad_graph_size = stuff[-1]
            flat_graph_state = stuff[:-1]
            gstate = GraphState.unflatten_from_const_size(flat_graph_state)

            # If necessary, update node state
            if self.node_state_update:
                gstate = self.node_state_updater.process(gstate, input_repr)

            # If necessary, propagate node state
            if self.intermediate_propagate != 0:
                gstate = self.intermediate_propagator.process_multiple(gstate, self.intermediate_propagate)

            # Propose and vote on new nodes
            gstate = self.new_node_adder.process(gstate, input_repr, self.new_nodes_per_iter)

            # Update edge state
            gstate = self.edge_state_updater.process(gstate, input_repr)

            return gstate.flatten_to_const_size(pad_graph_size)

        pad_graph_size = n_sentences * self.new_nodes_per_iter
        outputs_info = GraphState.create_empty(n_batch, self.node_state_size, self.edge_state_size).flatten_to_const_size(pad_graph_size)
        prepped_input = input_reprs.dimshuffle([1,0,2])
        all_flat_gstates = theano.scan(_scan_fn, sequences=[prepped_input], outputs_info=outputs_info, non_sequences=[pad_graph_size])
        final_flat_gstate = [x[:-1, ...] for x in all_flat_gstates]
        final_gstate = GraphState.unflatten_from_const_size(final_flat_gstate)

        aggregated_repr = self.aggregator.process(final_gstate) # shape (n_batch, output_repr_size)
        final_output = self.output_processor.process(aggregated_repr) # shape (n_batch, ?, num_output_words)

        # correct_output: shape (n_batch, ?, num_output_words)
        correct_output = T.ftensor3()

        if self.output_format == ModelOutputFormat.subset:
            elemwise_loss = T.nnet.binary_crossentropy(final_output, correct_output)
            full_loss = T.sum(elemwise_loss)
        else:
            flat_final_output = final_output.reshape([-1, self.num_output_words])
            flat_correct_output = correct_output.reshape([-1, self.num_output_words])
            timewise_loss = T.nnet.categorical_crossentropy(flat_final_output, flat_correct_output)
            full_loss = T.sum(timewise_loss)

        loss = full_loss/n_batch

        adam_updates = Adam(loss, self.params)

        self.train_fn = theano.function([input_words, correct_output],
                                        loss,
                                        updates=adam_updates,
                                        allow_input_downcast=True)

        self.eval_fn = theano.function( [input_words, correct_output],
                                        loss,
                                        allow_input_downcast=True)

        self.test_fn = theano.function( [input_words],
                                        [loss] + all_flat_gstates,
                                        allow_input_downcast=True)



