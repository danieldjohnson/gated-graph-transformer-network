import os
import pickle
import argparse
import shutil

import model
import babi_train
import babi_graph_parse
from util import *

def main(task_fn, output_format_str, state_width, dynamic_nodes, mutable_nodes, propagate_intermediate, outputdir, num_updates, batch_size, resume, resume_auto, visualize, validation):
    output_format = model.ModelOutputFormat[output_format_str]

    prepped_stories = babi_graph_parse.prepare_stories(babi_graph_parse.get_stories(task_fn), dynamic_nodes)
    sentence_length, new_nodes_per_iter, buckets, wordlist, anslist, graph_node_list, graph_edge_list, bucketed = prepped_stories
    eff_anslist = babi_train.get_effective_answer_words(anslist, output_format)

    if validation is None:
        validation_buckets = None
    else:
        validation_buckets = babi_parse.prepare_stories(babi_parse.get_stories(validation))[-1]

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
                    dynamic_nodes=dynamic_nodes,
                    nodes_mutable=mutable_nodes,
                    intermediate_propagate=(5 if propagate_intermediate else 0),
                    setup=True)

    if resume_auto:
        paramfile = os.path.join(outputdir,'final_params.p')
        with open(os.path.join(outputdir,'data.csv')) as f:
            for line in f:
                pass
            lastline = line
            start_idx = lastline.split(',')[0]
        print("Automatically resuming from {} after iteration {}.".format(paramfile, start_idx))
        resume = (start_idx, paramfile)

    if resume is not None:
        start_idx, paramfile = resume
        start_idx = int(start_idx)
        set_params(m.params, pickle.load(open(paramfile, "rb")))
    else:
        start_idx = 0

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    if visualize:
        print("Starting to visualize...")
        babi_train.visualize(m, bucketed, wordlist, eff_anslist, output_format, outputdir)
        print("Wrote visualization files to {}.".format(outputdir))
    else:
        print("Starting to train...")
        babi_train.train(m, bucketed, len(eff_anslist), output_format, num_updates, outputdir, start_idx, batch_size, validation_buckets)
        pickle.dump( m.params, open( os.path.join(outputdir, "final_params.p"), "wb" ) )

parser = argparse.ArgumentParser(description='Train a graph memory network model.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('task_fn', help="Filename of the task to load")
parser.add_argument('output_format_str', choices=[x.name for x in model.ModelOutputFormat], help="Output format for the task")
parser.add_argument('state_width', type=int, help="Width of node state")
parser.add_argument('--mutable-nodes', action="store_true", help="Make nodes mutable")
parser.add_argument('--dynamic-nodes', action="store_true", help="Create nodes after each sentence. (Otherwise, create unique nodes at the beginning)")
parser.add_argument('--propagate-intermediate', action="store_true", help="Run a propagation step after each sentence")
parser.add_argument('--outputdir', default="output", help="Directory to save output in")
parser.add_argument('--num-updates', default="10000", type=int, help="How many iterations to train")
parser.add_argument('--batch-size', default="10", type=int, help="Batch size to use")
parser.add_argument('--validation', metavar="VALIDATION_FILE", default=None, help="Filename of validation tasks")
parser.add_argument('--visualize', action="store_true", help="Visualise current state instead of training")
resume_group = parser.add_mutually_exclusive_group()
resume_group.add_argument('--resume', nargs=2, metavar=('TIMESTEP', 'PARAMFILE'), default=None, help='Where to restore from: timestep, and file to load')
resume_group.add_argument('--resume-auto', action='store_true', help='Automatically restore from a previous run using output directory')

if __name__ == '__main__':
    np.set_printoptions(linewidth=shutil.get_terminal_size((80, 20)).columns)
    args = vars(parser.parse_args())
    main(**args)
