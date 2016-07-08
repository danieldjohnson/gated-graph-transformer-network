import theano
import theano.tensor as T
import theano.sandbox.rng_mrg
import numpy as np

from enum import Enum
import itertools
import transformation_modules as tfms
from graph_state import GraphStateSpec, GraphState 
from adam import Adam

class ModelOutputFormat( Enum ):
    category = 1
    subset = 2
    sequence = 3

class Model( object ):
    """
    Implements the gated graph memory network model. 
    """

    def __init__(self, num_input_words, num_output_words, num_node_ids, node_state_size, num_edge_types, input_repr_size, output_repr_size, propose_repr_size, propagate_repr_size, new_nodes_per_iter, output_format, final_propagate, dynamic_nodes=True, nodes_mutable=True, intermediate_propagate=0, unroll_loop=0, dropout_keep=1, setup=True):
        """
        Parameters:
            num_input_words: How many possible words in the input
            num_output_words: How many possible words in the output
            num_node_ids: Id size (number of unique ids) for nodes
            node_state_size: State size for nodes
            num_edge_types: Number of unique edge types
            input_repr_size: Width of the intermediate input representation given to the network
            output_repr_size: Width of the intermediate output representation produced by the network
            propose_repr_size: Width of the indermediate new-node proposal representation
            propagate_repr_size: Width of the intermediate propagation representation
            new_nodes_per_iter: How many nodes to add at each sentence iteration
            output_format: Member of ModelOutputFormat, giving the format of the output
            final_propagate: How many steps to propagate info for each input sentence
            intermediate_propagate: How many steps to propagate info for each input sentence
            dynamic_nodes: Whether to dynamically create nodes as sentences are read. If false,
                a node with each id will be created at task start
            nodes_mutable: Whether nodes should update their state based on input
            unroll_loop: How many timesteps to unroll the main loop, or 0 to use scan
            dropout_keep: Probability to keep a node in dropout
            setup: Whether or not to automatically set up the model
        """
        self.num_input_words = num_input_words
        self.num_output_words = num_output_words
        self.num_node_ids = num_node_ids
        self.node_state_size = node_state_size
        self.num_edge_types = num_edge_types
        self.input_repr_size = input_repr_size
        self.output_repr_size = output_repr_size
        self.propose_repr_size = propose_repr_size
        self.propagate_repr_size = propagate_repr_size
        self.new_nodes_per_iter = new_nodes_per_iter
        self.output_format = output_format
        self.final_propagate = final_propagate
        self.intermediate_propagate = intermediate_propagate
        self.dynamic_nodes = dynamic_nodes
        self.nodes_mutable = nodes_mutable
        self.unroll_loop = unroll_loop
        self.dropout_keep = np.float32(dropout_keep)

        graphspec = GraphStateSpec(num_node_ids, node_state_size, num_edge_types)

        self.parameterized = []

        self.input_transformer = tfms.InputSequenceTransformation(num_input_words, input_repr_size)
        self.parameterized.append(self.input_transformer)

        if nodes_mutable:
            self.node_state_updater = tfms.NodeStateUpdateTransformation(input_repr_size, propose_repr_size, graphspec)
            self.parameterized.append(self.node_state_updater)

        if intermediate_propagate != 0:
            self.intermediate_propagator = tfms.PropagationTransformation(propagate_repr_size, graphspec, T.tanh)
            self.parameterized.append(self.intermediate_propagator)

        if self.dynamic_nodes:
            self.new_node_adder = tfms.NewNodesTransformation(input_repr_size, self.propose_repr_size, graphspec)
            self.parameterized.append(self.new_node_adder)

        self.edge_state_updater = tfms.EdgeStateUpdateTransformation(input_repr_size, graphspec)
        self.parameterized.append(self.edge_state_updater)

        self.query_node_state_updater = tfms.NodeStateUpdateTransformation(input_repr_size, graphspec)
        self.parameterized.append(self.query_node_state_updater)

        self.final_propagator = tfms.PropagationTransformation(propagate_repr_size, graphspec, T.tanh)
        self.parameterized.append(self.final_propagator)

        self.aggregator = tfms.AggregateRepresentationTransformation(output_repr_size, graphspec)
        self.parameterized.append(self.aggregator)

        self.srng = theano.sandbox.rng_mrg.MRG_RandomStreams(np.random.randint(0, 1024))

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
        # query_words: shape (n_batch, query_len)
        query_words = T.imatrix()
        # correct_output: shape (n_batch, ?, num_output_words)
        correct_output = T.ftensor3()

        # Get all dropout masks
        if self.nodes_mutable:
            node_state_updater_dropout = self.node_state_updater.get_dropout_masks(self.srng, self.dropout_keep)
        if self.intermediate_propagate != 0:
            intermediate_propagator_dropout = self.intermediate_propagator.get_dropout_masks(self.srng, self.dropout_keep)
        if self.dynamic_nodes:
            new_node_adder_dropout = self.new_node_adder.get_dropout_masks(self.srng, self.dropout_keep)
        edge_state_updater_dropout = self.edge_state_updater.get_dropout_masks(self.srng, self.dropout_keep)
        query_node_state_updater_dropout = self.query_node_state_updater.get_dropout_masks(self.srng, self.dropout_keep)
        final_propagator_dropout = self.final_propagator.get_dropout_masks(self.srng, self.dropout_keep)
        if self.output_format == ModelOutputFormat.sequence:
            output_processor_dropout = self.output_processor.get_dropout_masks(self.srng, self.dropout_keep)

        def _build(with_dropout):
            # Process each sentence, flattened to (?, sentence_len)
            flat_input_words = input_words.reshape([-1, sentence_len])
            flat_input_reprs = self.input_transformer.process(flat_input_words) # shape (?, input_repr_size)
            input_reprs = flat_input_reprs.reshape([n_batch, n_sentences, self.input_repr_size])

            query_repr = self.input_transformer.process(query_words)

            def _iter_fn(input_repr, gstate, all_dropouts):
                # If necessary, update node state
                if self.nodes_mutable:
                    num_dropout = self.node_state_updater.num_dropout_masks
                    if with_dropout:
                        cur_dropout = all_dropouts[:num_dropout]
                        all_dropouts = all_dropouts[num_dropout:]
                    else:
                        cur_dropout = None
                    gstate = self.node_state_updater.process(gstate, input_repr, cur_dropout)

                # If necessary, propagate node state
                if self.intermediate_propagate != 0:
                    num_dropout = self.intermediate_propagator.num_dropout_masks
                    if with_dropout:
                        cur_dropout = all_dropouts[:num_dropout]
                        all_dropouts = all_dropouts[num_dropout:]
                    else:
                        cur_dropout = None
                    gstate = self.intermediate_propagator.process_multiple(gstate, self.intermediate_propagate, cur_dropout)

                # Propose and vote on new nodes
                if self.dynamic_nodes:
                    num_dropout = self.new_node_adder.num_dropout_masks
                    if with_dropout:
                        cur_dropout = all_dropouts[:num_dropout]
                        all_dropouts = all_dropouts[num_dropout:]
                    else:
                        cur_dropout = None
                    gstate = self.new_node_adder.process(gstate, input_repr, self.new_nodes_per_iter, cur_dropout)

                # Update edge state
                num_dropout = self.edge_state_updater.num_dropout_masks
                if with_dropout:
                    cur_dropout = all_dropouts[:num_dropout]
                    all_dropouts = all_dropouts[num_dropout:]
                else:
                    cur_dropout = None
                gstate = self.edge_state_updater.process(gstate, input_repr, cur_dropout)
                return gstate

            # Scan over each sentence
            def _scan_fn(input_repr, *stuff): # (input_repr, *flat_graph_state, pad_graph_size, *dropouts)
                flat_graph_state = stuff[:GraphState.const_flattened_length()]
                pad_graph_size = stuff[GraphState.const_flattened_length()]
                dropouts = list(stuff[GraphState.const_flattened_length()+1:])
                gstate = GraphState.unflatten_from_const_size(flat_graph_state)

                gstate = _iter_fn(input_repr, gstate, dropouts)

                flat_gstate = gstate.flatten_to_const_size(pad_graph_size)
                return flat_gstate

            all_dropouts = (node_state_updater_dropout if self.nodes_mutable else []) \
                         + (intermediate_propagator_dropout if self.intermediate_propagate != 0 else []) \
                         + (new_node_adder_dropout if self.dynamic_nodes else []) \
                         + edge_state_updater_dropout
            if self.dynamic_nodes:
                initial_gstate = GraphState.create_empty(n_batch, self.num_node_ids, self.node_state_size, self.num_edge_types)
            else:
                initial_gstate = GraphState.create_full_unique(n_batch, self.num_node_ids, self.node_state_size, self.num_edge_types)
            if self.unroll_loop > 0:
                all_gstates = []
                cur_gstate = initial_gstate.with_updates(node_ids=T.opt.Assert("Maximum story length is {}".format(self.unroll_loop))(initial_gstate.node_ids, T.le(n_sentences,self.unroll_loop)))
                for i in range(self.unroll_loop):
                    condition = T.lt(i,n_sentences)
                    input_repr = input_reprs[:,i,:]
                    updated_gstate = _iter_fn(input_repr, cur_gstate, all_dropouts)
                    flat_updated_gstate = updated_gstate.flatten()
                    flat_cur_gstate = cur_gstate.flatten()
                    flat_new_gstate = [theano.ifelse.ifelse(condition, upd, cur) for upd,cur in zip(flat_updated_gstate, flat_cur_gstate)]
                    new_gstate = GraphState.unflatten(flat_new_gstate)
                    all_gstates.append(new_gstate)
                    cur_gstate = new_gstate
                final_gstate = cur_gstate
            else:
                # Account for all nodes, plus the extra padding node to prevent GPU unpleasantness
                if self.dynamic_nodes:
                    pad_graph_size = n_sentences * self.new_nodes_per_iter + 1
                else:
                    pad_graph_size = self.num_node_ids
                outputs_info = initial_gstate.flatten_to_const_size(pad_graph_size)
                prepped_input = input_reprs.dimshuffle([1,0,2])
                all_flat_gstates, _ = theano.scan(_scan_fn, sequences=[prepped_input], outputs_info=outputs_info, non_sequences=[pad_graph_size]+all_dropouts)
                final_flat_gstate = [x[-1] for x in all_flat_gstates]
                final_gstate = GraphState.unflatten_from_const_size(final_flat_gstate)

            query_gstate = self.query_node_state_updater.process(final_gstate, query_repr, query_node_state_updater_dropout if with_dropout else None)
            propagated_gstate = self.final_propagator.process_multiple(query_gstate, self.final_propagate, final_propagator_dropout if with_dropout else None)
            aggregated_repr = self.aggregator.process(propagated_gstate) # shape (n_batch, output_repr_size)
            
            max_seq_len = correct_output.shape[1]
            if self.output_format == ModelOutputFormat.sequence:
                final_output = self.output_processor.process(aggregated_repr, max_seq_len) # shape (n_batch, ?, num_output_words)
            else:
                final_output = self.output_processor.process(aggregated_repr)

            if self.output_format == ModelOutputFormat.subset:
                elemwise_loss = T.nnet.binary_crossentropy(final_output, correct_output)
                full_loss = T.sum(elemwise_loss)
            else:
                flat_final_output = final_output.reshape([-1, self.num_output_words])
                flat_correct_output = correct_output.reshape([-1, self.num_output_words])
                timewise_loss = T.nnet.categorical_crossentropy(flat_final_output, flat_correct_output)
                full_loss = T.sum(timewise_loss)

            loss = full_loss/T.cast(n_batch, 'float32')

            full_flat_gstates = [T.concatenate([a,T.shape_padleft(b),T.shape_padleft(c)],0).swapaxes(0,1)
                                    for a,b,c in zip(all_flat_gstates,
                                                     query_gstate.flatten(),
                                                     propagated_gstate.flatten())]
            return loss, final_output, full_flat_gstates, max_seq_len

        train_loss, _, _, _ = _build(self.dropout_keep != 1)
        adam_updates = Adam(train_loss, self.params)

        mode = theano.Mode().excluding("scanOp_pushout_output")
        self.train_fn = theano.function([input_words, query_words, correct_output],
                                        train_loss,
                                        updates=adam_updates,
                                        allow_input_downcast=True,
                                        mode=mode)

        eval_loss, final_output, full_flat_gstates, max_seq_len = _build(False)
        self.eval_fn = theano.function( [input_words, query_words, correct_output],
                                        eval_loss,
                                        allow_input_downcast=True,
                                        mode=mode)

        self.test_fn = theano.function( [input_words, query_words] + ([max_seq_len] if self.output_format == ModelOutputFormat.sequence else []),
                                        [final_output] + full_flat_gstates,
                                        allow_input_downcast=True,
                                        mode=mode)



