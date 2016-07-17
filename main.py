import os
import pickle
import argparse
import shutil
import math

import model
import babi_train
import babi_graph_parse
from babi_graph_parse import MetadataList, PreppedStory
from util import *

def helper_trim(bucketed, desired_total):
    """Trim bucketed fairly so that it has desired_total things total"""
    cur_total = sum(len(b) for b in bucketed)
    keep_frac = desired_total/cur_total
    if keep_frac > 1.0:
        print("WARNING: Asked to trim to {} items, but was already only {} items. Keeping original length.".format(desired_total, cur_total))
        return bucketed
    keep_amts = [math.floor(len(b) * keep_frac) for b in bucketed]
    tmp_total = sum(keep_amts)
    addtl_to_add = desired_total - tmp_total
    assert addtl_to_add >= 0
    keep_amts = [x + (1 if i < addtl_to_add else 0) for i,x in enumerate(keep_amts)]
    assert sum(keep_amts) == desired_total
    trimmed_bucketed = [b[:amt] for b,amt in zip(bucketed, keep_amts)]
    return trimmed_bucketed

def main(task_dir, output_format_str, state_width, dynamic_nodes, mutable_nodes, wipe_node_state, direct_reference, propagate_intermediate, train_with_graph, train_with_query, outputdir, num_updates, batch_size, resume, resume_auto, visualize, debugtest, validation, evaluate_accuracy, check_mode, stop_at_accuracy, restrict_dataset, train_save_params, batch_adjust, set_exit_status):
    output_format = model.ModelOutputFormat[output_format_str]

    with open(os.path.join(task_dir,'metadata.p'),'rb') as f:
        metadata = pickle.load(f)
    with open(os.path.join(task_dir,'file_list.p'),'rb') as f:
        bucketed = pickle.load(f)
    if restrict_dataset is not None:
        bucketed = helper_trim(bucketed, restrict_dataset)

    sentence_length, new_nodes_per_iter, bucket_sizes, wordlist, anslist, graph_node_list, graph_edge_list = metadata
    eff_anslist = babi_train.get_effective_answer_words(anslist, output_format)

    if validation is None:
        validation_buckets = None
    else:
        with open(os.path.join(validation,'metadata.p'),'rb') as f:
            validation_metadata = pickle.load(f)
        with open(os.path.join(validation,'file_list.p'),'rb') as f:
            validation_buckets = pickle.load(f)
        validation_bucket_sizes = validation_metadata[2]

    if direct_reference:
        word_node_mapping = {wi:ni for wi,word in enumerate(wordlist)
                                    for ni,node in enumerate(graph_node_list)
                                    if word == node}
    else:
        word_node_mapping = {}

    m = model.Model(num_input_words=len(wordlist),
                    num_output_words=len(eff_anslist),
                    num_node_ids=len(graph_node_list),
                    node_state_size=state_width,
                    num_edge_types=len(graph_edge_list),
                    input_repr_size=100,
                    output_repr_size=100,
                    propose_repr_size=50,
                    propagate_repr_size=50,
                    new_nodes_per_iter=new_nodes_per_iter,
                    output_format=output_format,
                    final_propagate=5,
                    word_node_mapping=word_node_mapping,
                    dynamic_nodes=dynamic_nodes,
                    nodes_mutable=mutable_nodes,
                    wipe_node_state=wipe_node_state,
                    intermediate_propagate=(5 if propagate_intermediate else 0),
                    best_node_match_only=True,
                    train_with_graph=train_with_graph,
                    train_with_query=train_with_query,
                    setup=True,
                    check_mode=check_mode)

    if resume_auto:
        paramfile = os.path.join(outputdir,'final_params.p')
        if os.path.isfile(paramfile):
            with open(os.path.join(outputdir,'data.csv')) as f:
                for line in f:
                    pass
                lastline = line
                start_idx = lastline.split(',')[0]
            print("Automatically resuming from {} after iteration {}.".format(paramfile, start_idx))
            resume = (start_idx, paramfile)
        else:
            print("Didn't find anything to resume. Starting from the beginning...")

    if resume is not None:
        start_idx, paramfile = resume
        start_idx = int(start_idx)
        load_params(m.params, open(paramfile, "rb") )
    else:
        start_idx = 0

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    if visualize is not False:
        if visualize is True:
            source = bucketed
        else:
            bucket, story = visualize
            source = [[bucketed[bucket][story]]]
        print("Starting to visualize...")
        babi_train.visualize(m, source, wordlist, eff_anslist, output_format, outputdir)
        print("Wrote visualization files to {}.".format(outputdir))
    elif evaluate_accuracy:
        print("Evaluating accuracy...")
        acc = babi_train.test_accuracy(m, bucketed, bucket_sizes, len(eff_anslist), output_format, batch_size)
        print("Obtained accuracy of {}".format(acc))
    elif debugtest:
        print("Starting debug test...")
        babi_train.visualize(m, bucketed, wordlist, eff_anslist, output_format, outputdir, debugmode=True)
        print("Wrote visualization files to {}.".format(outputdir))
    else:
        print("Starting to train...")
        status = babi_train.train(m, bucketed, bucket_sizes, len(eff_anslist), output_format, num_updates, outputdir, start_idx, batch_size, validation_buckets, validation_bucket_sizes, stop_at_accuracy, train_save_params, batch_adjust)
        save_params(m.params, open( os.path.join(outputdir, "final_params.p"), "wb" ) )
        if set_exit_status:
            sys.exit(status.value)

parser = argparse.ArgumentParser(description='Train a graph memory network model.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('task_dir', help="Parsed directory for the task to load")
parser.add_argument('output_format_str', choices=[x.name for x in model.ModelOutputFormat], help="Output format for the task")
parser.add_argument('state_width', type=int, help="Width of node state")
parser.add_argument('--mutable-nodes', action="store_true", help="Make nodes mutable")
parser.add_argument('--wipe-node-state', action="store_true", help="Wipe node state before the query")
parser.add_argument('--direct-reference', action="store_true", help="Use direct reference for input, based on node names")
parser.add_argument('--dynamic-nodes', action="store_true", help="Create nodes after each sentence. (Otherwise, create unique nodes at the beginning)")
parser.add_argument('--propagate-intermediate', action="store_true", help="Run a propagation step after each sentence")
parser.add_argument('--no-graph', dest='train_with_graph', action="store_false", help="Don't train using graph supervision")
parser.add_argument('--no-query', dest='train_with_query', action="store_false", help="Don't train using query supervision")
parser.add_argument('--outputdir', default="output", help="Directory to save output in")
parser.add_argument('--num-updates', default="10000", type=int, help="How many iterations to train")
parser.add_argument('--batch-size', default="10", type=int, help="Batch size to use")
parser.add_argument('--restrict-dataset', metavar="NUM_STORIES", type=int, default=None, help="Restrict size of dataset to this")
parser.add_argument('--final-params-only', action="store_false", dest="train_save_params", help="Don't save parameters while training, only at the end.")
parser.add_argument('--validation', metavar="VALIDATION_DIR", default=None, help="Parsed directory of validation tasks")
parser.add_argument('--check-nan', dest="check_mode", action="store_const", const="nan", help="Check for NaN. Slows execution")
parser.add_argument('--check-debug', dest="check_mode", action="store_const", const="debug", help="Debug mode. Slows execution")
parser.add_argument('--visualize', nargs="?", const=True, default=False, type=lambda s:[int(x) for x in s.split(',')], help="Visualise current state instead of training. Optional parameter to fix ")
parser.add_argument('--debugtest', action="store_true", help="Debug the training state")
parser.add_argument('--evaluate-accuracy', action="store_true", help="Evaluate accuracy of model")
parser.add_argument('--stop-at-accuracy', type=float, default=None, help="Stop training once it reaches this accuracy on validation set")
parser.add_argument('--batch-adjust', type=int, default=None, help="If set, ensure that size of edge matrix does not exceed this")
parser.add_argument('--set-exit-status', action="store_true", help="Give info about training status in the exit status")
resume_group = parser.add_mutually_exclusive_group()
resume_group.add_argument('--resume', nargs=2, metavar=('TIMESTEP', 'PARAMFILE'), default=None, help='Where to restore from: timestep, and file to load')
resume_group.add_argument('--resume-auto', action='store_true', help='Automatically restore from a previous run using output directory')

if __name__ == '__main__':
    np.set_printoptions(linewidth=shutil.get_terminal_size((80, 20)).columns)
    args = vars(parser.parse_args())
    main(**args)
