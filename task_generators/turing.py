import argparse
import random
import json
import sys
import generator_tools

def make_turing_machine_rules(n_states, n_symbols):
    the_rules = [   [   (random.randrange(n_symbols), random.randrange(n_states), random.choice('LNR'))
                        for symbol in range(n_symbols)]
                    for state in range(n_states)]
    return the_rules

def encode_turing_machine_rules(rules, starting_state=None, story=None):
    if story is None:
        story = generator_tools.Story()
    graph = story.graph
    if starting_state is None:
        starting_state = random.choice(len(rules))
    the_edges = [(cstate, read, write, nstate, direc)
                    for (cstate, stuff) in enumerate(rules)
                    for (read, (write, nstate, direc)) in enumerate(stuff)]
    random.shuffle(the_edges)
    for cstate, read, write, nstate, direc in the_edges:
        source = graph.make_unique('state_{}'.format(cstate))
        dest = graph.make_unique('state_{}'.format(nstate))
        edge_type = "rule_{}_{}_{}".format(read,write,direc)
        source[edge_type] = dest
        story.add_line("rule {} {} {} {} {}".format(source.type, read, write, dest.type, direc))
    head = graph.make_unique('head')

    head.state = graph.make_unique('state_{}'.format(starting_state))
    story.add_line("start {}".format(head.state.type))
    return story

def encode_turing_machine_process(rules, starting_state, iptlist, process_len, head_index=0, story=None, update_state=False):
    if story is None:
        story = generator_tools.Story()
    graph = story.graph
    last_input = None
    cells = []
    for i,symbol in enumerate(iptlist):
        cell = graph.make('cell')
        cell.left = last_input
        cell.value = graph.make_unique('symbol_{}'.format(symbol))
        cells.append(cell)
        last_input = cell
        if head_index == i:
            head = graph.make_unique('head')
            head.cell = cell
            story.add_line("input {} head".format(cell.value.type))
        else:
            story.add_line("input {}".format(cell.value.type))

    cstate = starting_state
    cell_values = iptlist[:]
    for _ in range(process_len):
        cell = cells[head_index]
        read = cell_values[head_index]
        write, nstate, direc = rules[cstate][read]
        cell.value = graph.make_unique('symbol_{}'.format(write))
        cstate = nstate
        if update_state:
            head.state = graph.make_unique('state_{}'.format(nstate))

        if direc == "L":
            if head_index == 0:
                newcell = graph.make('cell')
                cells.insert(0, newcell)
                cells[1].left = newcell
                newcell.value = graph.make_unique('symbol_{}'.format(0))
                cell_values.insert(0, 0)
                head_index += 1
            head_index -= 1
            head.cell = cells[head_index]
        elif direc == "R":
            if head_index == len(cells)-1:
                newcell = graph.make('cell')
                cells.append(newcell)
                newcell.left = cells[-2]
                newcell.value = graph.make_unique('symbol_{}'.format(0))
                cell_values.append(0)
            head_index += 1
            head.cell = cells[head_index]
        story.add_line('[RUN]')
    story.no_query()
    return story

def generate_universal(num_seqs, num_states, num_symbols, input_len, run_len):
    result = []
    for _ in range(num_seqs):
        rules = make_turing_machine_rules(num_states, num_symbols)
        start_state = random.randrange(num_states)
        input_list = [random.choice(range(num_symbols)) for _ in range(input_len)]
        head_index = random.randrange(input_len)
        story = encode_turing_machine_rules(rules, start_state)
        story = encode_turing_machine_process(rules, start_state, input_list, run_len, head_index, story, True)
        result.extend(story.lines)
    return "\n".join(result)+"\n"

def main(num_seqs, num_states, num_symbols, input_len, run_len, file):
    generated = generate_universal(num_seqs, num_states, num_symbols, input_len, run_len)
    file.write(generated)

parser = argparse.ArgumentParser(description='Generate a universal turing machine task')
parser.add_argument("file", nargs="?", default=sys.stdout, type=argparse.FileType('w'), help="Output file")
parser.add_argument("--num-states", type=int, default=4, help="Number of states")
parser.add_argument("--num-symbols", type=int, default=4, help="Number of symbols")
parser.add_argument("--input-len", type=int, default=5, help="Length of input")
parser.add_argument("--run-len", type=int, default=10, help="How many steps to simulate")
parser.add_argument("--num-seqs", type=int, default=1, help="Number of sequences to generate")

if __name__ == '__main__':
    args = vars(parser.parse_args())
    main(**args)




