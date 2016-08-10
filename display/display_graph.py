import numpy as np
import os
from IPython.display import Javascript
import json
import itertools
import pickle

from IPython.core.display import HTML

from .tolcolormap import cm_rainbow

def prep_graph_display(states, options={}):
    clean_states = [x.tolist() for x in states]
    nstr, nid, nstate, estr = states
    flat_nid = nid.reshape([-1,nid.shape[-1]])
    flat_estr = estr.reshape([-1,estr.shape[-1]])
    flat_estr = flat_estr / (np.linalg.norm(flat_estr, axis=1, keepdims=True) + 1e-8)

    num_unique_colors = nid.shape[-1] + estr.shape[-1]
    id_denom = max(nid.shape[-1] - 1, 1)
    id_project_mat = np.array([list(cm_rainbow(i/id_denom)[:3]) for i in range(0,nid.shape[-1])])
    estr_denom = estr.shape[-1]
    estr_project_mat = np.array([list(cm_rainbow((i+0.37)/estr_denom)[:3]) for i in range(estr.shape[-1])])
    node_colors = np.dot(flat_nid, id_project_mat)
    edge_colors = np.dot(flat_estr, estr_project_mat)

    colormap = {
        "node_id": node_colors.reshape(nid.shape[:-1] + (3,)).tolist(),
        "edge_type": edge_colors.reshape(estr.shape[:-1] + (3,)).tolist(),
    }

    return json.dumps({
        "states":clean_states,
        "colormap":colormap,
        "options":options
    })

def graph_display(states, options={}):
    stuff = prep_graph_display(states,options)
    return Javascript("var tmp={}; window.nonint_next = window._graph_display(tmp.states, tmp.colormap, element[0], 0, tmp.options);".format(stuff))

def noninteractive_next():
    return Javascript("window.nonint_next()")

def setup_graph_display():
    with open(os.path.join(os.path.dirname(__file__), "display_graph.js"), 'r') as f:
        JS_SETUP_STRING = f.read()
    return Javascript(JS_SETUP_STRING)

def main(visdir):
    results = []
    has_answer = os.path.isfile("{}/result_{}.npy".format(visdir,4))
    the_range = range(1,5) if has_answer else range(4)
    results = [np.load("{}/result_{}.npy".format(visdir,i)) for i in the_range]
    import importlib.machinery
    try:
        options_mod = importlib.machinery.SourceFileLoader('options',os.path.join(visdir,"options.py")).load_module()
        options = options_mod.options
    except FileNotFoundError:
        options = {}
    print(prep_graph_display(results,options))

import argparse
parser = argparse.ArgumentParser(description='Convert a visualization to JSON format')
parser.add_argument("visdir", help="Directory to visualization files")

if __name__ == '__main__':
    args = vars(parser.parse_args())
    main(**args)
