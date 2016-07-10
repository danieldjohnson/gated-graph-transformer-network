import argparse
import random
import json
import sys

def all_ngrams(seq, ngram_size):
    for i in range(len(seq)+1-ngram_size):
        yield tuple(seq[i:i+ngram_size])

def ngram_next_map(seq, ngram_size):
    the_map = {}
    for ngram in all_ngrams(seq, ngram_size+1):
        key = ngram[:-1]
        val = ngram[-1]
        if key in the_map and the_map[key] != val:
            # Don't want keys that appear twice
            the_map[key] = None
        else:
            the_map[key] = val
    return {k:v for k,v in the_map.items() if v is not None}

ITEM_PTR = "$ITEM$"
def generate(num_seqs, seq_length, ngram_size, symbols):
    assert ITEM_PTR not in symbols
    assert seq_length > ngram_size
    result = []
    for _ in range(num_seqs):
        while True: #just in case we don't find a good query
            story = []
            last_ptr = None
            values = []
            nodes = []
            edges = []
            for i in range(seq_length):
                # Choose next number
                next_item = random.choice(symbols)
                if not next_item in nodes:
                    nodes.append(next_item)
                cur_ptr = ITEM_PTR + "#" + str(i)
                nodes.append(cur_ptr)
                if last_ptr is not None:
                    edges.append({"from":last_ptr,"to":cur_ptr,"type":"next"})
                edges.append({"from":cur_ptr,"to":next_item,"type":"value"})
                last_ptr = cur_ptr
                values.append(next_item)
                graph_str = json.dumps({
                    "nodes":nodes,
                    "edges":edges,
                })
                story.append("{} {}={}".format(i+1, next_item, graph_str))
            possible_queries = ngram_next_map(values, ngram_size)
            if len(possible_queries) > 0:
                key, val = random.choice(list(possible_queries.items()))
                story.append("{} {}?\t{}".format(seq_length+1, ' '.join(key), val))
                result.extend(story)
                break
    return "\n".join(result)+"\n"

def main(num_seqs, seq_length, ngram_size, file):
    generated = generate(num_seqs, seq_length, ngram_size, [str(x) for x in range(10)])
    file.write(generated)

parser = argparse.ArgumentParser(description='Generate an ngrams task')
parser.add_argument("file", nargs="?", default=sys.stdout, type=argparse.FileType('w'), help="Output file")
parser.add_argument("--ngram-size", type=int, default=3, help="Size of ngrams")
parser.add_argument("--num-seqs", type=int, default=1, help="Number of sequences to generate")
parser.add_argument("--seq-length", type=int, default=10, help="Length of sequences to generate")

if __name__ == '__main__':
    args = vars(parser.parse_args())
    main(**args)
