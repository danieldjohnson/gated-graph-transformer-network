import model
import util
import pickle
import os
import argparse
import sys

def main(cache_dir):
    files_list = list(os.listdir(cache_dir))
    for file in files_list:
        full_filename = os.path.join(cache_dir, file)
        if os.path.isfile(full_filename):
            print("Processing {}".format(full_filename))
            m, stored_kwargs = pickle.load(open(full_filename, 'rb'))
            updated_kwargs = util.get_compatible_kwargs(model.Model, stored_kwargs)

            model_hash = util.object_hash(updated_kwargs)
            print("New hash -> " + model_hash)
            model_filename = os.path.join(cache_dir, "model_{}.p".format(model_hash))
            sys.setrecursionlimit(100000)
            pickle.dump((m,updated_kwargs), open(model_filename,'wb'), protocol=pickle.HIGHEST_PROTOCOL)

            os.remove(full_filename)

parser = argparse.ArgumentParser(description='Update a model cache directory.')
parser.add_argument('cache_dir', help="Directory of cached models to update")

if __name__ == '__main__':
    args = vars(parser.parse_args())
    main(**args)
