import run_harness
import argparse

def main(tasks_dir, output_dir):
    base_params = " ".join([
        "20",
        "--mutable-nodes",
        "--dynamic-nodes",
        "--num-updates 10000",
        "--batch-size 100",
        "--final-params-only",
        "--stop-at-accuracy 0.95",
        "--batch-adjust 28000000"])

    intermediate_propagate_tasks = {2,3,5,7,8}

    output_types = [
        "category", # [1]='WhereIsActor',
        "category", # [2]='WhereIsObject',
        "category", # [3]='WhereWasObject',
        "category", # [4]='IsDir',
        "category", # [5]='WhoWhatGave',
        "category", # [6]='IsActorThere',
        "category", # [7]='Counting',
        "set",      # [8]='Listing',
        "category", # [9]='Negation',
        "category", # [10]='Indefinite',
        "category", # [11]='BasicCoreference',
        "category", # [12]='Conjunction',
        "category", # [13]='CompoundCoreference',
        "category", # [14]='Time',
        "category", # [15]='Deduction',
        "category", # [16]='Induction',
        "category", # [17]='PositionalReasoning',
        "category", # [18]='Size',
        "sequence", # [19]='PathFinding',
        "category", # [20]='Motivations'
    ]

    restrict_sizes=[50,100,250,500,1000]

    specs = [run_harness.TaskSpec(  "task_{}".format(task_i),
                                    str(rsize) + ("-direct" if direct_ref else ""),
                                    "--restrict-dataset {} {}".format(rsize, ("--direct-reference" if direct_ref else "")))
                for task_i, output_type in zip(range(1,21),output_types)
                for rsize in restrict_sizes
                for direct_ref in (True, False)]

    run_harness.run(tasks_dir, output_dir, base_params, specs)

parser = argparse.ArgumentParser(description="Train all bAbI tasks.")
parser.add_argument('tasks_dir', help="Directory with tasks")
parser.add_argument('output_dir', help="Directory to save output to")

if __name__ == '__main__':
    args = vars(parser.parse_args())
    main(**args)

