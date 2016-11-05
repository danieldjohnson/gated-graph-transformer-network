import argparse
import random
import json
import sys
import graph_tools

def build_sequence(forth_sequence, run_steps=0):
    story = graph_tools.Story()
    graph = story.graph

    # Start
    n_pc = graph.make('pc')
    n_start = graph.make('c_START')
    n_pc.executing = n_start
    n_head = n_start
    next_cmd_edge = "next_cmd"
    scope_stack = [n_start]
    story.add_line("[START]")

    # Compiling
    for command in forth_sequence.split(' '):
        basic_cmds = ["NOP", "ZERO", "INC", "DEC", "DUP", "SWAP", "NOT", "POP", "HALT"]
        if command in basic_cmds:
            n_new = graph.make("c_{}".format(command))
            n_head[next_cmd_edge] = n_new
            n_head = n_new
            next_cmd_edge = "next_cmd"
        elif command == "IF":
            n_if = graph.make("c_IF")
            n_head[next_cmd_edge] = n_if
            n_head = n_if
            scope_stack[-1].next_scope = n_if
            scope_stack.append(n_if)
            next_cmd_edge = "next_if_true"
        elif command == "ELSE":
            n_if = scope_stack[-1]
            assert "c_IF" == n_if.type
            assert "c_IF" != n_head.type
            n_head.scope_end_cmd = n_if
            n_head = n_if
            next_cmd_edge = "next_if_false"
        elif command == "THEN":
            n_if = scope_stack.pop()
            scope_stack[-1].next_scope = None
            assert "c_IF" == n_if.type
            assert "c_IF" != n_head.type
            n_head.scope_end_cmd = n_if
            n_head = n_if
            next_cmd_edge = "next_then"
        elif command == "WHILE":
            n_while = graph.make("c_WHILE")
            n_head[next_cmd_edge] = n_while
            n_head = n_while
            scope_stack[-1].next_scope = n_while
            scope_stack.append(n_while)
            next_cmd_edge = "next_if_true"
        elif command == "REPEAT":
            n_while = scope_stack.pop()
            scope_stack[-1].next_scope = None
            assert "c_WHILE" == n_while.type
            assert "c_WHILE" != n_head.type
            n_head.scope_end_cmd = n_while
            n_head = n_while
        story.add_line(command)
    assert len(scope_stack) == 1

    # Running
    data_stack = []
    is_returning_to_if = False
    for i in range(run_steps):
        command = n_pc.executing.identifier[2:]
        if command == "IF" or command == "WHILE":
            assert len(data_stack) > 0
            if is_returning_to_if:
                n_pc.executing = n_pc.executing.next_then
                is_returning_to_if = False
            elif data_stack[-1].value is not None:
                n_pc.executing = n_pc.executing.next_if_true
            elif n_pc.executing.next_if_false is not None:
                n_pc.executing = n_pc.executing.next_if_false
            else:
                n_pc.executing = n_pc.executing.next_then
        elif command == "HALT":
            pass
        else:
            if command == "NOP":
                pass
            elif command == "ZERO":
                n_stacknode = graph.make("stacknode")
                if len(data_stack) > 0:
                    n_stacknode.prev = data_stack[-1]
                data_stack.append(n_stacknode)
                n_pc.stack_top = n_stacknode
            elif command == "INC":
                assert len(data_stack) > 0
                n_stacknode = data_stack[-1]
                n_counter = graph.make("counter")
                n_counter.successor = n_stacknode.value
                n_stacknode.value = n_counter
            elif command == "DEC":
                assert len(data_stack) > 0
                n_stacknode = data_stack[-1]
                if n_stacknode.value is not None:
                    n_stacknode.value = n_stacknode.value.successor
            elif command == "DUP":
                n_stacknode = graph.make("stacknode")
                if len(data_stack) > 0:
                    n_stacknode.prev = data_stack[-1]
                    n_stacknode.value = n_stacknode.prev.value
                data_stack.append(n_stacknode)
                n_pc.stack_top = n_stacknode
            elif command == "SWAP":
                assert len(data_stack) >= 2
                n_node1, n_node2 = data_stack[-2:]
                data_stack[-2:] = n_node2, n_node1
                n_node1.prev, n_node2.prev = n_node2.prev, n_node1.prev
            elif command == "POP":
                assert len(data_stack) > 0
                n_stacknode = data_stack.pop()
                n_stacknode.prev = None
                n_pc.stack_top = data_stack[-1]
            elif command == "NOT":
                assert len(data_stack) > 0
                n_stack_top = data_stack[-1]
                n_stacknode = graph.make("stacknode")
                n_stacknode.prev = data_stack[-1]
                if n_stack_top.value is None:
                    n_counter = graph.make("counter")
                    n_stacknode.value = n_counter
                data_stack.append(n_stacknode)
                n_pc.stack_top = n_stacknode
            if n_pc.executing.next_cmd is not None:
                n_pc.executing = n_pc.executing.next_cmd
            else:
                if n_pc.executing.scope_end_cmd.type == "c_IF":
                    is_returning_to_if = True
                n_pc.executing = n_pc.executing.scope_end_cmd
        assert n_pc.executing is not None
        story.add_line("[RUN]")

def _build_forth_string(max_len, stacklen=0):
    if max_len == 0:
        return [], stacklen
    chances = {
        "NOP":5,
        "ZERO":10,
        "INC":10 if stacklen>0 else 0,
        "DEC":3 if stacklen>0 else 0,
        "DUP":5 if stacklen>0 else 0,
        "SWAP":7 if stacklen>=2 else 0,
        "POP":3 if stacklen>0 else 0,
        "NOT":3 if stacklen>0 else 0,
        "HALT":2,
        "IF_THEN": 5 if max_len >=3 and stacklen>0 else 0,
        "IF_ELSE_THEN": 5 if max_len >=5 and stacklen>0 else 0,
        "WHILE_REPEAT": 10 if max_len >=3 and stacklen>0 else 0,
    }
    stack_deltas = {
        "ZERO":1,
        "DUP":1,
        "POP":-1,
    }
    chance_sum = sum(v for k,v in chances.items())
    while True:
        val = random.randrange(chance_sum)
        for cmd, chance in chances.items():
            val -= chance
            if val < 0:
                chosen_command = cmd
                break
        if chosen_command == "IF_THEN":
            tot_allocation = max_len - 2
            true_allocation = random.randrange(1, tot_allocation+1)
            then_allocation = tot_allocation - true_allocation

            true_cmds, true_stacklen = _build_forth_string(true_allocation, stacklen)
            next_stacklen = min(stacklen, true_stacklen)
            then_cmds, final_stacklen = _build_forth_string(then_allocation, next_stacklen)
            return (["IF"] + true_cmds + ["THEN"] + then_cmds), final_stacklen

        elif chosen_command == "IF_ELSE_THEN":
            tot_allocation = max_len - 3
            cond_allocation = random.randrange(2, tot_allocation+1)
            then_allocation = tot_allocation - cond_allocation
            true_allocation = random.randrange(1, cond_allocation)
            false_allocation = cond_allocation - true_allocation

            true_cmds, true_stacklen = _build_forth_string(true_allocation, stacklen)
            false_cmds, false_stacklen = _build_forth_string(false_allocation, stacklen)
            next_stacklen = min(true_stacklen, false_stacklen)
            then_cmds, final_stacklen = _build_forth_string(then_allocation, next_stacklen)

            return (["IF"] + true_cmds \
                   + ["ELSE"] + false_cmds \
                   + ["THEN"] + then_cmds), final_stacklen
        elif chosen_command == "WHILE_REPEAT":
            tot_allocation = max_len - 2
            true_allocation = random.randrange(1, tot_allocation+1)
            then_allocation = tot_allocation - true_allocation

            while True:
                true_cmds, true_stacklen = _build_forth_string(true_allocation, stacklen)
                if true_stacklen >= stacklen:
                    break
            then_cmds, final_stacklen = _build_forth_string(then_allocation, stacklen)
            return (["WHILE"] + true_cmds + ["REPEAT"] + then_cmds), final_stacklen
        else:
            if chosen_command in stack_deltas:
                next_stacklen = stacklen + stack_deltas[chosen_command]
            else:
                next_stacklen = stacklen
            rest_cmds, rest_stacklen = _build_forth_string(max_len-1, next_stacklen)
            return [chosen_command] + rest_cmds, rest_stacklen

def build_forth_string(max_len):
    return " ".join(_build_forth_string(max_len)[0] + ["HALT"])

def generate(num_seqs, seq_length):
    for _ in range(num_seqs):
        forth_string = build_forth_string(seq_length)
        print(forth_string)



