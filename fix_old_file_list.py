import pickle
import os
import argparse
import sys

def main(task_dir, dry_run=False):
    with open(os.path.join(task_dir,'file_list.p'),'rb') as f:
        bucketed = pickle.load(f)
    if dry_run:
        print("Got {} (for example)".format(bucketed[0][0]))
    bucketed = [['./bucket_' + x.split('bucket_')[1] for x in b] for b in bucketed]
    if dry_run:
        print("Converting to {} (for example)".format(bucketed[0][0]))
        print("Will resolve to {} (for example)".format(os.path.normpath(os.path.join(task_dir,bucketed[0][0]))))
    else:
        with open(os.path.join(task_dir,'file_list.p'),'wb') as f:
            pickle.dump(bucketed, f)

parser = argparse.ArgumentParser(description='Fix the file list of a parsed directory.')
parser.add_argument('task_dir', help="Directory of parsed files")
parser.add_argument('--dry-run', action="store_true", help="Don't overwrite files")

if __name__ == '__main__':
    args = vars(parser.parse_args())
    main(**args)
