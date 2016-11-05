import argparse
import os
import numpy as np
import ggtnn_graph_parse
from ggtnn_graph_parse import PreppedStory
import gzip
import pickle

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

def main(storyfile, outputdir):
    
    with gzip.open(storyfile,'rb') as f:
        story, sents, query, ans = pickle.load(f)

    with open(os.path.join(outputdir,'story.txt'),'w') as f:
        f.write("{}\n{}\n{}".format("\n".join(" ".join(s) for s in sents), " ".join(query), " ".join(ans)))

    results = convert(story)
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    for i,res in enumerate(results):
        np.save(os.path.join(outputdir,'result_{}.npy'.format(i)), res)

parser = argparse.ArgumentParser(description='Convert a story to graph')
parser.add_argument("storyfile", help="Story filename")
parser.add_argument("outputdir", help="Output directory")

if __name__ == '__main__':
    args = vars(parser.parse_args())
    main(**args)


