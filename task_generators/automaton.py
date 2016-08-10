import argparse
import random
import json
import sys

def simulate(cells, rules):
    assert rules[(0,0,0)] == 0
    old_cells = [0,0] + cells + [0,0]
    new_cells = []
    for i in range(len(cells)+2):
        cur_block = tuple(old_cells[i:i+3])
        new_cells.append(rules[cur_block])
    return new_cells

def int_to_bintuple(val,width):
    val = tuple(int(x) for x in bin(val)[2:])
    while len(val) < width:
        val = (0,) + val
    return val

def decode_rules(rule_idx):
    keys = [int_to_bintuple(i,3) for i in reversed(range(8))]
    values = int_to_bintuple(rule_idx,8)
    return dict(zip(keys,values))

def generate(num_seqs, init_len, run_len, rule_idx, start_with=None):
    assert init_len > 0
    rules = decode_rules(rule_idx)
    result = []
    for _ in range(num_seqs):
        story = []
        cell_ptrs = []
        cell_values = []
        nodes = []
        connect_edges = []
        value_edges = []
        if start_with is None:
            val_sequence = [random.choice([0,1]) for _ in range(init_len)]
        else:
            val_sequence = [int(x) for x in start_with]
        for i,val in enumerate(val_sequence):
            cell_values.append(val)
            val_node = str(val)
            if val_node not in nodes:
                nodes.append(val_node)
            cell_node = "cell_init#"+str(i)
            nodes.append(cell_node)
            value_edges.append({"type":"value","from":cell_node,"to":val_node})
            if len(cell_ptrs) > 0:
                connect_edges.append({"type":"next_r","from":cell_ptrs[-1],"to":cell_node})
            cell_ptrs.append(cell_node)

            graph_str = json.dumps({
                "nodes":nodes,
                "edges":connect_edges + value_edges,
            })
            story.append("init {}={}".format(val,graph_str))
        for i in range(run_len):
            new_cell_values = simulate(cell_values, rules)
            cell_left = "cell_left#"+str(i)
            cell_right = "cell_right#"+str(i)
            connect_edges.append({"type":"next_r","from":cell_left,"to":cell_ptrs[0]})
            connect_edges.append({"type":"next_r","from":cell_ptrs[-1],"to":cell_right})
            nodes.extend([cell_left,cell_right])
            cell_ptrs = [cell_left] + cell_ptrs + [cell_right]
            value_edges = []
            for cell_ptr,val in zip(cell_ptrs,new_cell_values):
                val_node = str(val)
                if val_node not in nodes:
                    nodes.append(val_node)
                value_edges.append({"type":"value","from":cell_ptr,"to":val_node})
            cell_values = new_cell_values
            graph_str = json.dumps({
                "nodes":nodes,
                "edges":connect_edges + value_edges,
            })
            story.append("simulate={}".format(graph_str))
        story.append("\t")
        result.extend(["{} {}".format(i+1,s) for i,s in enumerate(story)])
    return "\n".join(result)+"\n"

def main(num_seqs, init_len, run_len, rule_idx, file, start_with):
    generated = generate(num_seqs, init_len, run_len, rule_idx, start_with)
    file.write(generated)

parser = argparse.ArgumentParser(description='Generate an ngrams task')
parser.add_argument("rule_idx", type=int, help="Which automaton rule to use")
parser.add_argument("file", nargs="?", default=sys.stdout, type=argparse.FileType('w'), help="Output file")
parser.add_argument("--num-seqs", type=int, default=1, help="Number of sequences to generate")
parser.add_argument("--init-len", type=int, default=5, help="Length of initial cells")
parser.add_argument("--run-len", type=int, default=5, help="Number of simulate steps")
parser.add_argument("--start-with", default=None, help="Start with this exact input")

if __name__ == '__main__':
    args = vars(parser.parse_args())
    main(**args)









