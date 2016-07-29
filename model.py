import theano
import theano.tensor as T
import theano.sandbox.rng_mrg
import numpy as np

import util

from enum import Enum
import itertools
import transformation_modules as tfms
from graph_state import GraphStateSpec, GraphState 
from adam import Adam

from theano.compile.nanguardmode import NanGuardMode
from theano.compile.debugmode import DebugMode

class ModelOutputFormat( Enum ):
    category = 1
    subset = 2
    sequence = 3

class Model( object ):
    """
    Implements the gated graph memory network model. 
    """

    def __init__(self, num_input_words, num_output_words, num_node_ids, node_state_size, num_edge_types, input_repr_size, output_repr_size, propose_repr_size, propagate_repr_size, new_nodes_per_iter, output_format, final_propagate, word_node_mapping={},  dynamic_nodes=True, nodes_mutable=True, wipe_node_state=True, best_node_match_only=True, intermediate_propagate=0, use_old_aggregate=False, train_with_graph=True, train_with_query=True, setup=True, check_mode=None):
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
            word_node_mapping: Dictionary mapping word ids to node ids for direct reference in input
            best_node_match_only: If the network should only train on the ordering with the
                best match
            intermediate_propagate: How many steps to propagate info for each input sentence
            use_old_aggregate: Should it use the old (sofmax) activation
            dynamic_nodes: Whether to dynamically create nodes as sentences are read. If false,
                a node with each id will be created at task start
            nodes_mutable: Whether nodes should update their state based on input
            wipe_node_state: Whether to wipe node state at the query
            train_with_graph: If True, use the graph to train. Otherwise ignore the graph
            train_with_query: If True, use the query to train. Otherwise ignore the query
            setup: Whether or not to automatically set up the model
            check_mode: If 'nan', run in NaNGuardMode. If 'debug', run in DebugMode
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
        self.word_node_mapping = word_node_mapping
        self.best_node_match_only = best_node_match_only
        self.intermediate_propagate = intermediate_propagate
        self.use_old_aggregate = use_old_aggregate
        self.dynamic_nodes = dynamic_nodes
        self.nodes_mutable = nodes_mutable
        self.wipe_node_state = wipe_node_state
        self.train_with_graph = train_with_graph
        self.train_with_query = train_with_query
        self.check_mode = check_mode

        AggregateRepresentationTransformation = tfms.AggregateRepresentationTransformationSoftmax \
                                                if use_old_aggregate \
                                                else tfms.AggregateRepresentationTransformation

        graphspec = GraphStateSpec(num_node_ids, node_state_size, num_edge_types)

        self.parameterized = []

        self.input_transformer = tfms.InputSequenceDirectTransformation(num_input_words, num_node_ids, word_node_mapping, input_repr_size)
        self.parameterized.append(self.input_transformer)

        if nodes_mutable:
            self.node_state_updater = tfms.NodeStateUpdateTransformation(input_repr_size, graphspec)
            self.parameterized.append(self.node_state_updater)

        if len(self.word_node_mapping) > 0:
            self.direct_reference_updater = tfms.DirectReferenceUpdateTransformation(input_repr_size, graphspec)
            self.parameterized.append(self.direct_reference_updater)

        if intermediate_propagate != 0:
            self.intermediate_propagator = tfms.PropagationTransformation(propagate_repr_size, graphspec, T.tanh)
            self.parameterized.append(self.intermediate_propagator)

        if self.dynamic_nodes:
            self.new_node_adder = tfms.NewNodesInformTransformation(input_repr_size, self.propose_repr_size, self.propose_repr_size, graphspec, use_old_aggregate)
            self.parameterized.append(self.new_node_adder)

        self.edge_state_updater = tfms.EdgeStateUpdateTransformation(input_repr_size, graphspec)
        self.parameterized.append(self.edge_state_updater)

        if self.train_with_query:
            self.query_node_state_updater = tfms.NodeStateUpdateTransformation(input_repr_size, graphspec)
            self.parameterized.append(self.query_node_state_updater)

            if len(self.word_node_mapping) > 0:
                self.query_direct_reference_updater = tfms.DirectReferenceUpdateTransformation(input_repr_size, graphspec)
                self.parameterized.append(self.query_direct_reference_updater)
            
            self.final_propagator = tfms.PropagationTransformation(propagate_repr_size, graphspec, T.tanh)
            self.parameterized.append(self.final_propagator)

            self.aggregator = AggregateRepresentationTransformation(output_repr_size, graphspec)
            self.parameterized.append(self.aggregator)

            assert output_format in ModelOutputFormat, "Invalid output format {}".format(output_format)
            if output_format == ModelOutputFormat.category:
                self.output_processor = tfms.OutputCategoryTransformation(output_repr_size, num_output_words)
            elif output_format == ModelOutputFormat.subset:
                self.output_processor = tfms.OutputSetTransformation(output_repr_size, num_output_words)
            elif output_format == ModelOutputFormat.sequence:
                self.output_processor = tfms.OutputSequenceTransformation(output_repr_size, output_repr_size, num_output_words)
            self.parameterized.append(self.output_processor)

        self.srng = theano.sandbox.rng_mrg.MRG_RandomStreams(np.random.randint(0, 1024))

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

        # graph_num_new_nodes: shape(n_batch, n_sentence)
        graph_num_new_nodes = T.imatrix()
        # graph_new_node_strengths: shape(n_batch, n_sentence, new_nodes_per_iter)
        graph_new_node_strengths =  T.ftensor3()
        # graph_new_node_ids: shape(n_batch, n_sentence, new_nodes_per_iter, num_node_ids)
        graph_new_node_ids = T.ftensor4()
        # graph_new_edges: shape(n_batch, n_sentence, pad_graph_size, pad_graph_size, num_edge_types)
        graph_new_edges = T.TensorType('floatX', (False,)*5)()

        def _build(with_correct_graph, snap_to_best):
            info = {}
            # Process each sentence, flattened to (?, sentence_len)
            flat_input_words = input_words.reshape([-1, sentence_len])
            flat_input_reprs, flat_ref_matrices = self.input_transformer.process(flat_input_words)
            # flat_input_reprs of shape (?, input_repr_size)
            # flat_ref_matrices of shape (?, num_node_ids, input_repr_size)
            input_reprs = flat_input_reprs.reshape([n_batch, n_sentences, self.input_repr_size])
            ref_matrices = flat_ref_matrices.reshape([n_batch, n_sentences, self.num_node_ids, self.input_repr_size])

            query_repr, query_ref_matrix = self.input_transformer.process(query_words)

            def _iter_fn(input_repr, ref_matrix, gstate, correct_num_new_nodes=None, correct_new_strengths=None, correct_new_node_ids=None, correct_edges=None):
                # If necessary, update node state
                if self.nodes_mutable:
                    gstate = self.node_state_updater.process(gstate, input_repr)

                if len(self.word_node_mapping) > 0:
                    gstate = self.direct_reference_updater.process(gstate, ref_matrix)

                # If necessary, propagate node state
                if self.intermediate_propagate != 0:
                    gstate = self.intermediate_propagator.process_multiple(gstate, self.intermediate_propagate)

                node_loss = None
                # Propose and vote on new nodes
                if self.dynamic_nodes:
                    new_strengths, new_ids = self.new_node_adder.get_candidates(gstate, input_repr, self.new_nodes_per_iter)
                    # new_strengths and correct_new_strengths are of shape (n_batch, new_nodes_per_iter)
                    # new_ids and correct_new_node_ids are of shape (n_batch, new_nodes_per_iter, num_node_ids)
                    if with_correct_graph:
                        perm_idxs = np.array(list(itertools.permutations(range(self.new_nodes_per_iter))))
                        permuted_correct_str = correct_new_strengths[:,perm_idxs]
                        permuted_correct_ids = correct_new_node_ids[:,perm_idxs]
                        # due to advanced indexing, we should have shape (n_batch, permutation, new_nodes_per_iter, num_node_ids)
                        ext_new_str = T.shape_padaxis(new_strengths,1)
                        ext_new_ids = T.shape_padaxis(new_ids,1)
                        strength_ll = permuted_correct_str * T.log(ext_new_str + util.EPSILON) + (1-permuted_correct_str) * T.log(1-ext_new_str + util.EPSILON)
                        ids_ll = permuted_correct_ids * T.log(ext_new_ids  + util.EPSILON)
                        reduced_perm_lls = T.sum(strength_ll, axis=2) + T.sum(ids_ll, axis=[2,3])
                        if self.best_node_match_only:
                            node_loss = -T.max(reduced_perm_lls, 1)
                        else:
                            full_ll = util.reduce_log_sum(reduced_perm_lls, 1)
                            # Note that some of these permutations are identical, since we likely did not add the maximum
                            # amount of nodes. Thus we will have added repeated elements here.
                            # We have log(x+x+...+x) = log(kx), where k is the repetition factor and x is the probability we want
                            # log(kx) = log(k) + log(x)
                            # Our repetition factor k is given by (new_nodes_per_iter - correct_num_new_nodes)!
                            # Recall that n! = gamma(n+1)
                            # so log(x) = log(kx) - log(gamma(k+1))
                            log_rep_factor = T.gammaln(T.cast(self.new_nodes_per_iter - correct_num_new_nodes + 1, 'floatX'))
                            scaled_ll = full_ll - log_rep_factor
                            node_loss = -scaled_ll
                        # now substitute in the correct nodes
                        gstate = gstate.with_additional_nodes(correct_new_strengths, correct_new_node_ids)
                    elif snap_to_best:
                        snapped_strengths = util.independent_best(new_strengths)
                        snapped_ids = util.categorical_best(new_ids)
                        gstate = gstate.with_additional_nodes(snapped_strengths, snapped_ids)
                    else:
                        gstate = gstate.with_additional_nodes(new_strengths, new_ids)


                # Update edge state
                gstate = self.edge_state_updater.process(gstate, input_repr)
                if with_correct_graph:
                    cropped_correct_edges = correct_edges[:,:gstate.n_nodes,:gstate.n_nodes,:]
                    edge_lls = cropped_correct_edges * T.log(gstate.edge_strengths + util.EPSILON) + (1-cropped_correct_edges) * T.log(1-gstate.edge_strengths + util.EPSILON)
                    # edge_lls currently penalizes for edges connected to nodes that do not exist
                    # we do not want it to do this, so we mask it with node strengths
                    mask_src = util.shape_padaxes(gstate.node_strengths,[2,3])
                    mask_dest = util.shape_padaxes(gstate.node_strengths,[1,3])
                    masked_edge_lls = edge_lls * mask_src * mask_dest
                    edge_loss = -T.sum(masked_edge_lls, axis=[1,2,3])
                    gstate = gstate.with_updates(edge_strengths=cropped_correct_edges)
                    return gstate, node_loss, edge_loss
                elif snap_to_best:
                    snapped_edges = util.independent_best(gstate.edge_strengths)
                    gstate = gstate.with_updates(edge_strengths=snapped_edges)
                    return gstate
                else:
                    return gstate

            # Scan over each sentence
            def _scan_fn(input_repr, *stuff): # (input_repr, [ref_matrix?], [*correct_graph_stuff?], *flat_graph_state, pad_graph_size)
                stuff = list(stuff)

                if len(self.word_node_mapping) > 0:
                    ref_matrix = stuff[0]
                    stuff = stuff[1:]
                else:
                    ref_matrix = None

                if with_correct_graph:
                    c_num_new_nodes, c_new_strengths, c_new_node_ids, c_edges = stuff[:4]
                    stuff = stuff[4:]

                flat_graph_state = stuff[:-1]
                pad_graph_size = stuff[-1]
                gstate = GraphState.unflatten_from_const_size(flat_graph_state)

                if with_correct_graph:
                    gstate, node_loss, edge_loss = _iter_fn(input_repr, ref_matrix, gstate, c_num_new_nodes, c_new_strengths, c_new_node_ids, c_edges)
                else:
                    gstate = _iter_fn(input_repr, ref_matrix, gstate)

                retvals = gstate.flatten_to_const_size(pad_graph_size)
                if with_correct_graph:
                    if self.dynamic_nodes:
                        retvals.append(node_loss)
                    retvals.append(edge_loss)
                return retvals

            if self.dynamic_nodes:
                initial_gstate = GraphState.create_empty(n_batch, self.num_node_ids, self.node_state_size, self.num_edge_types)
            else:
                initial_gstate = GraphState.create_full_unique(n_batch, self.num_node_ids, self.node_state_size, self.num_edge_types)

            # Account for all nodes, plus the extra padding node to prevent GPU unpleasantness
            if self.dynamic_nodes:
                pad_graph_size = n_sentences * self.new_nodes_per_iter + 1
            else:
                pad_graph_size = self.num_node_ids
            outputs_info = initial_gstate.flatten_to_const_size(pad_graph_size)
            prepped_input = input_reprs.dimshuffle([1,0,2])
            sequences = [prepped_input]
            if len(self.word_node_mapping) > 0:
                sequences.append(ref_matrices.dimshuffle([1,0,2,3]))
            if with_correct_graph:
                sequences.append(graph_num_new_nodes.swapaxes(0,1))
                sequences.append(graph_new_node_strengths.swapaxes(0,1))
                sequences.append(graph_new_node_ids.swapaxes(0,1))
                sequences.append(graph_new_edges.swapaxes(0,1))

                if self.dynamic_nodes:
                    outputs_info.extend([None])
                outputs_info.extend([None])
            all_scan_out, _ = theano.scan(_scan_fn, sequences=sequences, outputs_info=outputs_info, non_sequences=[pad_graph_size])
            if with_correct_graph:
                if self.dynamic_nodes:
                    all_flat_gstates = all_scan_out[:-2]
                    node_loss, edge_loss = all_scan_out[-2:]
                    reduced_node_loss = T.sum(node_loss)/T.cast(n_batch, 'floatX')
                    reduced_edge_loss = T.sum(edge_loss)/T.cast(n_batch, 'floatX')
                    avg_graph_loss = (reduced_node_loss + reduced_edge_loss)/T.cast(input_words.shape[1], 'floatX')
                    info["node_loss"]=reduced_node_loss
                    info["edge_loss"]=reduced_edge_loss
                else:
                    all_flat_gstates = all_scan_out[:-1]
                    edge_loss = all_scan_out[-1]
                    reduced_edge_loss = T.sum(edge_loss)/T.cast(n_batch, 'floatX')
                    avg_graph_loss = reduced_edge_loss/T.cast(input_words.shape[1], 'floatX')
                    info["edge_loss"]=reduced_edge_loss
            else:
                all_flat_gstates = all_scan_out
            final_flat_gstate = [x[-1] for x in all_flat_gstates]
            final_gstate = GraphState.unflatten_from_const_size(final_flat_gstate)

            if self.train_with_query:
                if self.wipe_node_state:
                    final_gstate = final_gstate.with_updates(node_states=T.zeros_like(final_gstate.node_states))
                query_gstate = self.query_node_state_updater.process(final_gstate, query_repr)
                if len(self.word_node_mapping) > 0:
                    query_gstate = self.query_direct_reference_updater.process(query_gstate, query_ref_matrix)
                propagated_gstate = self.final_propagator.process_multiple(query_gstate, self.final_propagate)
                aggregated_repr = self.aggregator.process(propagated_gstate) # shape (n_batch, output_repr_size)
                
                max_seq_len = correct_output.shape[1]
                if self.output_format == ModelOutputFormat.sequence:
                    final_output = self.output_processor.process(aggregated_repr, max_seq_len) # shape (n_batch, ?, num_output_words)
                else:
                    final_output = self.output_processor.process(aggregated_repr)

                if snap_to_best:
                    final_output = self.output_processor.snap_to_best(final_output)

                if self.output_format == ModelOutputFormat.subset:
                    elemwise_loss = T.nnet.binary_crossentropy(final_output, correct_output)
                    query_loss = T.sum(elemwise_loss)
                else:
                    flat_final_output = final_output.reshape([-1, self.num_output_words])
                    flat_correct_output = correct_output.reshape([-1, self.num_output_words])
                    timewise_loss = T.nnet.categorical_crossentropy(flat_final_output, flat_correct_output)
                    query_loss = T.sum(timewise_loss)
                query_loss = query_loss/T.cast(n_batch, 'floatX')
                info["query_loss"] = query_loss
            else:
                final_output = T.zeros([])

            full_loss = np.array(0.0,np.float32)
            if with_correct_graph:
                full_loss = full_loss + avg_graph_loss
            if self.train_with_query:
                full_loss = full_loss + query_loss
            full_loss = T.opt.Assert("Full loss was nan!")(full_loss, T.invert(T.isnan(full_loss)))

            if self.train_with_query:
                full_flat_gstates = [T.concatenate([a,T.shape_padleft(b),T.shape_padleft(c)],0).swapaxes(0,1)
                                        for a,b,c in zip(all_flat_gstates[:-1],
                                                         query_gstate.flatten(),
                                                         propagated_gstate.flatten())]
            else:
                full_flat_gstates = [a.swapaxes(0,1) for a in all_flat_gstates[:-1]]
                max_seq_len = T.iscalar()
            return full_loss, final_output, full_flat_gstates, max_seq_len, info

        train_loss, _, full_flat_gstates, _, train_info = _build(self.train_with_graph, False)
        adam_updates = Adam(train_loss, self.params)

        self.info_keys = list(train_info.keys())

        if self.check_mode == 'nan':
            mode = NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
        elif self.check_mode == 'debug':
            mode = DebugMode()
        else:
            mode = theano.Mode()
        mode = mode.excluding("scanOp_pushout_output")
        self.train_fn = theano.function([input_words, query_words, correct_output, graph_num_new_nodes, graph_new_node_strengths, graph_new_node_ids, graph_new_edges],
                                        [train_loss]+list(train_info.values()),
                                        updates=adam_updates,
                                        allow_input_downcast=True,
                                        on_unused_input='ignore',
                                        mode=mode)

        self.eval_fn = theano.function( [input_words, query_words, correct_output, graph_num_new_nodes, graph_new_node_strengths, graph_new_node_ids, graph_new_edges],
                                        [train_loss]+list(train_info.values()),
                                        allow_input_downcast=True,
                                        on_unused_input='ignore',
                                        mode=mode)

        self.debug_test_fn = theano.function( [input_words, query_words, correct_output, graph_num_new_nodes, graph_new_node_strengths, graph_new_node_ids, graph_new_edges],
                                        full_flat_gstates,
                                        allow_input_downcast=True,
                                        on_unused_input='ignore',
                                        mode=mode)

        test_loss, final_output, full_flat_gstates, max_seq_len, _ = _build(False, False)
        self.fuzzy_test_fn = theano.function( [input_words, query_words] + ([max_seq_len] if self.output_format == ModelOutputFormat.sequence else []),
                                        [final_output] + full_flat_gstates,
                                        allow_input_downcast=True,
                                        on_unused_input='ignore',
                                        mode=mode)

        test_loss, final_output, full_flat_gstates, max_seq_len, _ = _build(False, True)
        self.snap_test_fn = theano.function( [input_words, query_words] + ([max_seq_len] if self.output_format == ModelOutputFormat.sequence else []),
                                        [final_output] + full_flat_gstates,
                                        allow_input_downcast=True,
                                        on_unused_input='ignore',
                                        mode=mode)

    def train(self, *args, **kwargs):
        stuff = self.train_fn(*args, **kwargs)
        loss = stuff[0]
        info = dict(zip(self.info_keys, stuff[1:]))
        return loss, info

    def eval(self, *args, **kwargs):
        stuff = self.eval_fn(*args, **kwargs)
        loss = stuff[0]
        info = dict(zip(self.info_keys, stuff[1:]))
        return loss, info

