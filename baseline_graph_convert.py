import collections
import random
import pickle
from ggtnn_graph_parse import *

# A baseline graph is formatted as a single sequence of tokens, with indices as follows:
# ...,RESERVED_CT: magic symbols
# [RESERVED_CT,RESERVED_CT+N): node id
# [RESERVED_CT+N, RESERVED_CT+N+E]: edge id

RESERVED_CT = 3
# SYM_STOP = 0
SYM_CLOSE = 0
SYM_PREV = 1
SYM_NEXT = 2
# SYM_ENDSTEPS = 3

def baseline_convert_graph(graphs, nodemap, edgemap, dynamic=True):
    N = len(nodemap)
    E = len(edgemap)

    descriptions = []
    processed_nodes = []
    number_per_type = collections.defaultdict(int)
    counter_map = {}
    def append_node(desc, node, relto=None):
        desc.append(RESERVED_CT + nodemap[get_unqualified_id(node)])
        if relto is not None:
            if get_unqualified_id(node) == get_unqualified_id(relto):
                diff = counter_map[node] - counter_map[relto]
            else:
                diff = counter_map[node]
            for _ in range(abs(diff)):
                desc.append(SYM_NEXT if diff>0 else SYM_PREV)
        # desc.append(SYM_ENDSTEPS)

    for g in graphs:
        active_nodes = g["nodes"]
        active_edges = g["edges"]

        new_nodes = [e for e in active_nodes if e not in processed_nodes]
        for n in new_nodes:
            ntype = get_unqualified_id(n)
            counter_map[n] = number_per_type[ntype]
            number_per_type[ntype] += 1

        sorted_nodes = sorted(active_nodes, key=lambda n:(nodemap[get_unqualified_id(n)],counter_map[n]))

        node_outgoing = collections.defaultdict(lambda:[])
        for edge in active_edges:
            node_outgoing[edge["from"]].append((edgemap[edge["type"]],edge["to"]))

        cur_desc = []
        for node in sorted_nodes:
            append_node(cur_desc, node)
            outgoing = node_outgoing[node]
            outgoing.sort(key=lambda et:(et[0],nodemap[get_unqualified_id(et[1])],counter_map[et[1]]))
            for typenum, dest in outgoing:
                cur_desc.append(RESERVED_CT + N + typenum)
                append_node(cur_desc, dest, relto=node)
            cur_desc.append(SYM_CLOSE)
        # cur_desc.append(SYM_STOP)

        processed_nodes.extend(new_nodes)
        descriptions.append(cur_desc)

    return descriptions

def baseline_convert_story(story, wordmap, answer_map, graph_node_map, graph_edge_map, dynamic=True):
    """
    Converts a story in format
        ([(sentence, graph)], [(index, question_arr, answer)])
    to a consolidated story in format
        (sentence_arr, [graph_arr_dict], [(index, question_arr, answer)])
    and also replaces words according to the input maps
    """
    sents_graphs, query, answer = story

    sentence_arr = [[wordmap[w] for w in s] for s,g in sents_graphs]
    graphs = baseline_convert_graph([g for s,g in sents_graphs], graph_node_map, graph_edge_map, dynamic)
    query_arr = [wordmap[w] for w in query]
    answer_arr = [answer_map[w] for w in answer]
    return (sentence_arr, graphs, query_arr, answer_arr)

# def baseline_bucket_stories(stories, buckets, wordmap, answer_map, graph_node_map, graph_edge_map, sentence_length, dynamic=True):
#     def process_story(s,bucket_len):
#         return baseline_convert_story(pad_story(s, bucket_len, sentence_length), wordmap, answer_map, graph_node_map, graph_edge_map, dynamic)
#     return [ [process_story(story,bmax) for story in stories if bstart < len(story[0]) <= bmax]
#                 for bstart, bmax in zip([0]+buckets,buckets)]

# def baseline_prepare_stories(stories, dynamic=True):
#     sentence_length = max(get_max_sentence_length(stories), get_max_query_length(stories))
#     buckets = get_buckets(stories)
#     wordlist, wordmap = get_wordlist(stories)
#     anslist, ansmap = get_answer_list(stories)

#     graph_node_list, graph_node_map, graph_edge_list, graph_edge_map = get_graph_lists(stories)
#     bucketed = baseline_bucket_stories(stories, buckets, wordmap, ansmap, graph_node_map, graph_edge_map, sentence_length, dynamic)
#     return sentence_length, new_nodes_per_iter, buckets, wordlist, anslist, graph_node_list, graph_edge_list, bucketed

def baseline_print_graph(descs, nodelist, edgelist, file=sys.stdout):
    parts = []
    for i,desc in enumerate(descs):
        parts.append("Step {} (length {})\n".format(i, len(desc)))
        for item in desc:
            if item == SYM_CLOSE: parts.append("; ")
            elif item == SYM_PREV: parts.append("-")
            elif item == SYM_NEXT: parts.append("+")
            elif (item-RESERVED_CT) < len(nodelist):
                parts.append(nodelist[item-RESERVED_CT])
            else:
                parts.append(" "+edgelist[item-RESERVED_CT-len(nodelist)]+"->")
        parts.append("\n\n")
    file.write("".join(parts).encode())

def baseline_encode_single_graph(desc, nodelist, edgelist):
    parts = []
    for item in desc:
        if item == SYM_CLOSE: parts.append(";")
        elif item == SYM_PREV: parts.append("-")
        elif item == SYM_NEXT: parts.append("+")
        elif (item-RESERVED_CT) < len(nodelist):
            parts.append(nodelist[item-RESERVED_CT])
        else:
            parts.append(edgelist[item-RESERVED_CT-len(nodelist)])
    return parts

def baseline_preprocess_stories(stories, savedir, dynamic=True, metadata_file=None):
    if metadata_file is None:
        metadata = generate_metadata(stories, dynamic)
    else:
        with open(metadata_file,'rb') as f:
            metadata = pickle.load(f)

    buckets = get_buckets(stories)
    sentence_length, new_nodes_per_iter, old_buckets, wordlist, anslist, graph_node_list, graph_edge_list = metadata
    metadata = metadata._replace(buckets=buckets)

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    with open(os.path.join(savedir,'metadata.p'),'wb') as f:
        pickle.dump(metadata, f)

    bucketed_files = [[] for _ in buckets]

    maxiptlen = 0
    maxoptlen = 0

    with open(os.path.join(savedir,'in.txt'),'wb') as infile:
        with open(os.path.join(savedir,'out.txt'),'wb') as outfile:

            lines = []
            for i,story in enumerate(stories):
                # bucket_idx, cur_bucket = next(((i,bmax) for (i,(bstart, bmax)) in enumerate(zip([0]+buckets,buckets))
                #                                 if bstart < len(story[0]) <= bmax), (None,None))
                # assert cur_bucket is not None, "Couldn't put story of length {} into buckets {}".format(len(story[0]), buckets)
                # bucket_dir = os.path.join(savedir, "bucket_{}".format(cur_bucket))
                # if not os.path.exists(bucket_dir):
                #     os.makedirs(bucket_dir)
                # story_fn = os.path.join(bucket_dir, "story_{}.pz".format(i))

                sents_graphs, query, answer = story
                sents = [s for s,g in sents_graphs]
                cvtd = baseline_convert_story(story, list_to_map(wordlist), list_to_map(anslist), list_to_map(graph_node_list), list_to_map(graph_edge_list), dynamic)
                graphs = cvtd[1]

                for sent, ingraph, outgraph in zip(sents, itertools.chain([[]], graphs), graphs):
                    iptwords = sent + [">>>>>"] + baseline_encode_single_graph(ingraph, graph_node_list, graph_edge_list)
                    optwords = baseline_encode_single_graph(outgraph, graph_node_list, graph_edge_list)
                    lines.append((" ".join(iptwords)," ".join(optwords)))
                    maxiptlen = max(maxiptlen, len(iptwords))
                    maxoptlen = max(maxoptlen, len(optwords))
                    # if len(optwords) > 800:
                    #     print(optwords, len(optwords))

                # prepped = PreppedStory(cvtd, sents, query, answer)

                # with gzip.open(story_fn, 'wb') as zf:
                #     pickle.dump(prepped, zf)

                # text_fn = os.path.join(bucket_dir, "out_{}.txt".format(i))
                # with open(text_fn, 'wb') as f:
                #     baseline_print_graph_machine(cvtd[1],graph_node_list,graph_edge_list,f)

                # bucketed_files[bucket_idx].append(os.path.relpath(story_fn, savedir))
                gc.collect() # we don't want to use too much memory, so try to clean it up
            random.shuffle(lines)
            for inline, outline in lines:
                infile.write((inline + "\n").encode())
                outfile.write((outline + "\n").encode())

    outputWordSet = set.union(set(graph_node_list), set(graph_edge_list), set([";","-","+"]))
    inputWordSet = set.union(set(wordlist), set([">>>>>"]), outputWordSet)

    def processVocabSet(s):
        d = {w:i+2 for i,w in enumerate(sorted(s))}
        d['<S>'] = 0
        d['<UNK>'] = 1
        d['</S>'] = len(d)
        return d

    outputVocab = processVocabSet(outputWordSet)
    inputVocab = processVocabSet(inputWordSet)

    with open(os.path.join(savedir,'inputVocab.pkl'),'wb') as f:
        pickle.dump(inputVocab,f,pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(savedir,'outputVocab.pkl'),'wb') as f:
        pickle.dump(outputVocab,f,pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(savedir,'options.txt'),'wb') as f:
        # gct = RESERVED_CT + len(graph_node_list) + len(graph_edge_list)
        f.write("input vocab: {}\n".format( len(inputVocab) ).encode())
        f.write("output vocab: {}\n".format( len(outputVocab) ).encode())
        f.write("input len: {}\n".format( maxiptlen ).encode())
        f.write("output len: {}\n".format( maxoptlen ).encode())

    # with open(os.path.join(savedir,'file_list.p'),'wb') as f:
    #     pickle.dump(bucketed_files, f)

def main(file, dynamic, metadata_file=None):
    stories = get_stories(file)
    dirname, ext = os.path.splitext(file)
    baseline_preprocess_stories(stories, dirname+"_baseline", dynamic, metadata_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse a graph file')
    parser.add_argument("file", help="Graph file to parse")
    parser.add_argument("--static", dest="dynamic", action="store_false", help="Don't use dynamic nodes")
    parser.add_argument("--metadata-file", default=None, help="Use this particular metadata file instead of building it from scratch")
    args = vars(parser.parse_args())
    main(**args)

