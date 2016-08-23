import run_harness
import argparse
import os

def main(tasks_dir, output_dir, excluding=[], run_sequential_set=False):
    base_params = " ".join([
        "20",
        "--mutable-nodes",
        "--dynamic-nodes",
        "--num-updates 3000",
        "--batch-size 100",
        "--final-params-only",
        "--learning-rate 0.002",
        "--save-params-interval 100",
        "--validation-interval 100",
        "--batch-adjust 28000000"])

    intermediate_propagate_tasks = {3,5}
    alt_sequential_set = {3,5}

    output_types = [
        "category", # [1]='WhereIsActor',
        "category", # [2]='WhereIsObject',
        "category", # [3]='WhereWasObject',
        "category", # [4]='IsDir',
        "category", # [5]='WhoWhatGave',
        "category", # [6]='IsActorThere',
        "category", # [7]='Counting',
        "subset",   # [8]='Listing',
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

    tasks_and_outputs = list(zip(range(1,21),output_types))
    if run_sequential_set:
        base_params = base_params + " --sequence-aggregate-repr"
        tasks_and_outputs = [tasks_and_outputs[x-1] for x in alt_sequential_set]
        intermediate_propagate_tasks = set()

    specs = [run_harness.TaskSpec(  "task_{}".format(task_i),
                                    str(rsize) + ("-direct" if direct_ref else ""),
                                    "{} --restrict-dataset {} --stop-at-accuracy {} {} {} {}".format(
                                        output_type,
                                        rsize,
                                        "1.0" if rsize==1000 else "0.95",
                                        "--propagate-intermediate" if task_i in intermediate_propagate_tasks else "",
                                        "" if rsize==1000 else "--stop-at-overfitting 5",
                                        ("--direct-reference" if direct_ref else "")))
                for rsize in reversed(restrict_sizes)
                for direct_ref in (True,False)
                for task_i, output_type in tasks_and_outputs]

    specs = [x for x in specs if x.task_name not in excluding]

    run_harness.run(tasks_dir, output_dir, base_params, specs)

parser = argparse.ArgumentParser(description="Train all bAbI tasks.")
parser.add_argument('tasks_dir', help="Directory with tasks")
parser.add_argument('output_dir', help="Directory to save output to")
parser.add_argument('--excluding', nargs='+', default=[], help="Tasks to exclude")
parser.add_argument('--run-sequential-set', action="store_true", help="Run tasks with sequential output instead, and only run tasks that need it")

if __name__ == '__main__':
    args = vars(parser.parse_args())
    main(**args)

