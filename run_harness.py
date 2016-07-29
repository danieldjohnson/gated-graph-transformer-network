import sys
import os
import subprocess
import shutil
import shlex
import collections
from babi_train import TrainExitStatus
from graceful_interrupt import GracefulInterruptHandler

TaskSpec = collections.namedtuple("TaskSpec", ["task_name", "variant_name", "run_params"])

def run(tasks_dir, output_dir, base_params, specs):
    base_params_split = shlex.split(base_params)
    for spec in specs:
        print("### Task {} ({}) ###".format(spec.task_name, spec.variant_name))
        run_params_split = shlex.split(spec.run_params)

        task_folder_train = os.path.join(tasks_dir, "{}_train".format(spec.task_name))
        if not os.path.isdir(task_folder_train):
            print("Train directory doesn't exist. Parsing text file...")
            textfile = task_folder_train + ".txt"
            subprocess.run(["python3","babi_graph_parse.py",textfile], check=True)

        task_folder_valid = os.path.join(tasks_dir, "{}_valid".format(spec.task_name))
        if not os.path.isdir(task_folder_valid):
            print("Validation directory doesn't exist. Parsing text file...")
            textfile = task_folder_valid + ".txt"
            subprocess.run(["python3","babi_graph_parse.py",textfile], check=True)

        task_output_dir = os.path.join(output_dir, spec.task_name, spec.variant_name)
        if not os.path.isdir(task_output_dir):
            os.makedirs(task_output_dir)

        completed_file = os.path.join(task_output_dir, "completed.txt")
        if os.path.exists(completed_file):
            print("Task is already completed! Skipping...")
            continue

        stdout_fn = os.path.join(task_output_dir, "stdout.txt")

        all_params = [task_folder_train] + run_params_split + base_params_split
        all_params.extend(["--outputdir", task_output_dir])
        all_params.extend(["--validation", task_folder_valid])
        all_params.extend(["--set-exit-status"])
        all_params.extend(["--resume-auto"])
        with open(stdout_fn, 'a', 1) as stdout_file:
            proc = subprocess.Popen(all_params, stdout=stdout_file, stderr=subprocess.STDOUT)
            with GracefulInterruptHandler():
                returncode = proc.wait()
        task_status = TrainExitStatus(returncode)

        if task_status == TrainExitStatus.accuracy_success:
            print("SUCCESS! Reached desired accuracy.")
            with open(completed_file,'w') as f:
                f.write("SUCCESS\n")
        elif task_status == TrainExitStatus.reached_update_limit:
            print("FAIL! Reached update limit without attaining desired accuracy.")
            with open(completed_file,'w') as f:
                f.write("FAIL_UPDATE_LIMIT\n")
        elif task_status == TrainExitStatus.overfitting:
            print("FAIL! Detected overfitting.")
            with open(completed_file,'w') as f:
                f.write("FAIL_OVERFITTING\n")
        elif task_status == TrainExitStatus.error:
            print("Got an error; skipping for now. See {} for details.".format(stdout_fn))
        elif task_status == TrainExitStatus.nan_loss:
            print("NaN loss detected; skipping for now.")
        elif task_status == TrainExitStatus.interrupted:
            print("Process was interrupted! Stopping now")
            break
