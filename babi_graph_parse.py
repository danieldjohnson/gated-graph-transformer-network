import os
import sys
import re
import collections
import numpy as np
import scipy
import json
import itertools

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return re.findall('(?:\w+)|\S',sent)

def parse_stories(lines):
    '''
    Parse stories provided in the bAbi tasks format, with knowledge graph.

    Returns a list of stories of the form:
        ([(sentence, graph)], [(index, question, answer)])
    '''
    data = []
    story = []
    for line in lines:
        if line[-1] == "\n":
            line = line[:-1]
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
            questions = []
        if '\t' in line:
            q, apre = line.split('\t')[:2]
            a = apre.split(',')
            q = tokenize(q)
            substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            line, graph = line.split('=', 1)
            sent = tokenize(line)
            graph_parsed = json.loads(graph)
            story.append((sent, graph_parsed))
    return data

def get_stories(taskname):
    with open(taskname, 'r') as f:
        lines = f.readlines()
    return parse_stories(lines)

def get_max_sentence_length(stories):
    return max((max((len(sentence) for (sentence, graph) in sents_graphs)) for (sents_graphs, query, answer) in stories))

def get_max_query_length(stories):
    return max((len(query) for (sents_graphs, query, answer) in stories))

def get_max_num_queries(stories):
    return max((len(queries) for (sents_graphs, query, answer) in stories))

def get_max_nodes_per_iter(stories):
    result = 0
    for (sents_graphs, query, answer) in stories:
        prev_nodes = set()
        for (sentence, graph) in sents_graphs:
            cur_nodes = set(graph["nodes"])
            new_nodes = len(cur_nodes - prev_nodes)
            if new_nodes > result:
                result = new_nodes
            prev_nodes = cur_nodes
    return result

def get_buckets(stories, max_ignore_unbatched=100, max_pad_amount=25):
    sentencecounts = [len(sents_graphs) for (sents_graphs, query, answer) in stories]
    countpairs = sorted(collections.Counter(sentencecounts).items())

    buckets = []
    smallest_left_val = 0
    num_unbatched = max_ignore_unbatched
    for val,ct in countpairs:
        num_unbatched += ct
        if val - smallest_left_val > max_pad_amount or num_unbatched > max_ignore_unbatched:
            buckets.append(val)
            smallest_left_val = val
            num_unbatched = 0
    if buckets[-1] != countpairs[-1][0]:
        buckets.append(countpairs[-1][0])

    return buckets

PAD_WORD = "<PAD>"

def get_wordlist(stories):
    words = [PAD_WORD] + sorted(list(set((word
        for (sents_graphs, query, answer) in stories
        for wordbag in itertools.chain((s for s,g in sents_graphs), [query])
        for word in wordbag ))))
    wordmap = {word:i for i,word in enumerate(words)}
    return words, wordmap

def get_answer_list(stories):
    words = sorted(list(set(word for (sents_graphs, query, answer) in stories for word in answer)))
    wordmap = {word:i for i,word in enumerate(words)}
    return words, wordmap

def pad_story(story, num_sentences, sentence_length):
    def pad(lst,dlen,pad):
        return lst + [pad]*(dlen - len(lst))
    
    sents_graphs, query, answer = story
    padded_sents_graphs = [(pad(s,sentence_length,PAD_WORD), g) for s,g in sents_graphs]
    padded_query = pad(query,sentence_length,PAD_WORD)

    sentgraph_padding = (pad([],sentence_length,PAD_WORD), padded_sents_graphs[-1][1])
    return (pad(padded_sents_graphs, num_sentences, sentgraph_padding), padded_query, answer)

def get_unqualified_id(s):
    return s.split("#")[0]

def get_graph_lists(stories):
    node_words = sorted(list(set(get_unqualified_id(node)
        for (sents_graphs, query, answer) in stories
        for sent,graph in sents_graphs
        for node in graph["nodes"])))
    nodemap = {word:i for i,word in enumerate(node_words)}
    edge_words = sorted(list(set(get_unqualified_id(edge["type"])
        for (sents_graphs, query, answer) in stories
        for sent,graph in sents_graphs
        for edge in graph["edges"])))
    edgemap = {word:i for i,word in enumerate(edge_words)}
    return node_words, nodemap, edge_words, edgemap

def convert_graph(graphs, nodemap, edgemap, new_nodes_per_iter, dynamic=True):
    num_node_ids = len(nodemap)
    num_edge_types = len(edgemap)

    full_size = len(graphs)*new_nodes_per_iter + 1

    prev_size = 1
    processed_nodes = []
    index_map = {}
    all_num_nodes = []
    all_node_ids = []
    all_node_strengths = []
    all_edges = []
    if not dynamic:
        processed_nodes = list(nodemap.keys())
        index_map = nodemap.copy()
        prev_size = num_node_ids
        full_size = prev_size
        new_nodes_per_iter = 0
    for g in graphs:
        active_nodes = g["nodes"]
        active_edges = g["edges"]

        new_nodes = [e for e in active_nodes if e not in processed_nodes]

        num_new_nodes = len(new_nodes)
        if not dynamic:
            assert num_new_nodes == 0, "Cannot create more nodes in non-dynamic mode!\n{}".format(graphs)
        
        new_node_strengths = np.zeros([new_nodes_per_iter], np.float32)
        new_node_strengths[:num_new_nodes] = 1.0

        new_node_ids = np.zeros([new_nodes_per_iter, num_node_ids], np.float32)
        for i, node in enumerate(new_nodes):
            new_node_ids[i,nodemap[get_unqualified_id(node)]] = 1.0
            index_map[node] = prev_size + i

        next_edges = np.zeros([full_size, full_size, num_edge_types])
        for edge in active_edges:
            next_edges[index_map[edge["from"]],
                       index_map[edge["to"]],
                       edgemap[get_unqualified_id(edge["type"])]] = 1.0

        processed_nodes.extend(new_nodes)
        prev_size += new_nodes_per_iter

        all_num_nodes.append(num_new_nodes)
        all_node_ids.append(new_node_ids)
        all_edges.append(next_edges)
        all_node_strengths.append(new_node_strengths)

    return np.stack(all_num_nodes), np.stack(all_node_strengths), np.stack(all_node_ids), np.stack(all_edges)

def convert_story(story, wordmap, answer_map, graph_node_map, graph_edge_map, new_nodes_per_iter, dynamic=True):
    """
    Converts a story in format
        ([(sentence, graph)], [(index, question_arr, answer)])
    to a consolidated story in format
        (sentence_arr, [graph_arr_dict], [(index, question_arr, answer)])
    and also replaces words according to the input maps
    """
    sents_graphs, query, answer = story

    sentence_arr = [[wordmap[w] for w in s] for s,g in sents_graphs]
    graphs = convert_graph([g for s,g in sents_graphs], graph_node_map, graph_edge_map, new_nodes_per_iter, dynamic)
    query_arr = [wordmap[w] for w in query]
    answer_arr = [answer_map[w] for w in answer]
    return (sentence_arr, graphs, query_arr, answer_arr)

def bucket_stories(stories, buckets, wordmap, answer_map, graph_node_map, graph_edge_map, sentence_length, new_nodes_per_iter, dynamic=True):
    def process_story(s,bucket_len):
        return convert_story(pad_story(s, bucket_len, sentence_length), wordmap, answer_map, graph_node_map, graph_edge_map, new_nodes_per_iter, dynamic)
    return [ [process_story(story,bmax) for story in stories if bstart < len(story[0]) <= bmax]
                for bstart, bmax in zip([0]+buckets,buckets)]

def prepare_stories(stories, dynamic=True):
    sentence_length = max(get_max_sentence_length(stories), get_max_query_length(stories))
    buckets = get_buckets(stories)
    wordlist, wordmap = get_wordlist(stories)
    anslist, ansmap = get_answer_list(stories)
    new_nodes_per_iter = get_max_nodes_per_iter(stories)

    graph_node_list, graph_node_map, graph_edge_list, graph_edge_map = get_graph_lists(stories)
    bucketed = bucket_stories(stories, buckets, wordmap, ansmap, graph_node_map, graph_edge_map, sentence_length, new_nodes_per_iter, dynamic)
    return sentence_length, new_nodes_per_iter, buckets, wordlist, anslist, graph_node_list, graph_edge_list, bucketed

def print_batch(story, wordlist, anslist, file=sys.stdout):
    sents, query, answer = story
    for batch,(s,q,a) in enumerate(zip(sents,query,answer)):
        file.write("Story {}\n".format(batch))
        for sent in s:
            file.write(" ".join([wordlist[word] for word in sent]) + "\n")
        file.write(" ".join(wordlist[word] for word in q) + "\n")
        file.write(" ".join(anslist[word] for word in a.nonzero()[1]) + "\n")
