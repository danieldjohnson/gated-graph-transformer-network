import argparse
import os
import numpy as np
import babi_graph_parse

def convert(story):
    # import pdb; pdb.set_trace()
    sentence_arr, graphs, query_arr, answer_arr = story
    node_id_w = graphs[2].shape[2]
    edge_type_w = graphs[3].shape[3]

    all_node_strengths = [np.zeros([1])]
    all_node_ids = [np.zeros([1,node_id_w])]
    for num_new_nodes, new_node_strengths, new_node_ids, _ in zip(*graphs):
        last_strengths = all_node_strengths[-1]
        last_ids = all_node_ids[-1]

        cur_strengths = np.concatenate([last_strengths, new_node_strengths], 0)
        cur_ids = np.concatenate([last_ids, new_node_ids], 0)

        all_node_strengths.append(cur_strengths)
        all_node_ids.append(cur_ids)

    all_edges = graphs[3]
    full_n_nodes = all_edges.shape[1]
    all_node_strengths = np.stack([np.pad(x, ((0, full_n_nodes-x.shape[0])), 'constant') for x in all_node_strengths[1:]])
    all_node_ids = np.stack([np.pad(x, ((0, full_n_nodes-x.shape[0]), (0, 0)), 'constant') for x in all_node_ids[1:]])
    all_node_states = np.zeros([len(all_node_strengths), full_n_nodes,0])

    return tuple(x[np.newaxis,...] for x in (all_node_strengths, all_node_ids, all_node_states, all_edges))

def main(task_fn, index1, index2, outputdir, dynamic=True):
    prepped_stories = babi_graph_parse.prepare_stories(babi_graph_parse.get_stories(task_fn), dynamic)
    sentence_length, new_nodes_per_iter, buckets, wordlist, anslist, graph_node_list, graph_edge_list, bucketed = prepped_stories
    print("buckets:", buckets)
    story = bucketed[index1][index2]
    results = convert(story)
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    for i,res in enumerate(results):
        print(res.shape)
        np.save(os.path.join(outputdir,'result_{}.npy'.format(i)), res)

parser = argparse.ArgumentParser(description='Generate an ngrams task')
parser.add_argument("task_fn", help="Task filename")
parser.add_argument("index1", type=int, help="Bucket index")
parser.add_argument("index2", type=int, help="Story index")
parser.add_argument("outputdir", help="Output directory")
parser.add_argument("--static", action="store_false", dest="dynamic", help="Don't use dynamic mode")

if __name__ == '__main__':
    args = vars(parser.parse_args())
    main(**args)


