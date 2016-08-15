from babi_graph_parse import MetadataList
from pprint import pprint
import pickle
import sys

def main(file):
    with open(file,'rb') as f:
        metadata = pickle.load(f)
    pprint(dict(metadata._asdict()))

if __name__ == '__main__':
    main(sys.argv[1])