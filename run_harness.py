import sys
import os
import subprocess
import shutil
import shlex
import collections
from train_exit_status import TrainExitStatus
from graceful_interrupt import GracefulInterruptHandler
from termcolor import colored

TaskSpec = collections.namedtuple("TaskSpec", ["task_name", "variant_name", "run_params"])

def run(tasks_dir, output_dir, base_params, specs, stop_on_error=False, skip_complete=False):
    base_params_split = shlex.split(base_params)
    for spec in specs:
        print(colored("### Task {} ({}) ###".format(spec.task_name, spec.variant_name), "yellow"))
        run_params_split = shlex.split(spec.run_params)

        task_folder_train = os.path.join(tasks_dir, "{}_train".format(spec.task_name))
        if not os.path.isdir(task_folder_train):
            print(colored("Train directory doesn't exist. Parsing text file...", attrs=["dark"]))
            textfile = task_folder_train + ".txt"
            subprocess.run(["python3","ggtnn_graph_parse.py",textfile], check=True)

        task_folder_valid = os.path.join(tasks_dir, "{}_valid".format(spec.task_name))
        if not os.path.isdir(task_folder_valid):
            print(colored("Validation directory doesn't exist. Parsing text file...", attrs=["dark"]))
            textfile = task_folder_valid + ".txt"
            try:
                subprocess.run(["python3","ggtnn_graph_parse.py",textfile,"--metadata-file",os.path.join(task_folder_train,"metadata.p")], check=True)
            except subprocess.CalledProcessError:
                print(colored("Could not parse validation set! Skipping. You may need to regenerate the training set.","magenta"))
                continue

        task_output_dir = os.path.join(output_dir, spec.task_name, spec.variant_name)
        if not os.path.isdir(task_output_dir):
            os.makedirs(task_output_dir)

        completed_file = os.path.join(task_output_dir, "completed.txt")
        if os.path.exists(completed_file):
            with open(completed_file,'r') as f:
                reason = f.readline().strip()
            reason = colored(reason, "green" if (reason == "SUCCESS") else "red" if ("FAIL" in reason) else "magenta")
            print("Task is already completed, with result {}. Skipping...".format(reason))
            continue

        stdout_fn = os.path.join(task_output_dir, "stdout.txt")

        all_params = ["python3", "-u", "main.py", task_folder_train] + run_params_split + base_params_split
        all_params.extend(["--outputdir", task_output_dir])
        all_params.extend(["--validation", task_folder_valid])
        all_params.extend(["--set-exit-status"])
        all_params.extend(["--resume-auto"])
        all_params.extend(["--autopickle", os.path.join(output_dir, "model_cache")])
        print("Running command: " + " ".join(all_params))
        with open(stdout_fn, 'a', 1) as stdout_file:
            proc = subprocess.Popen(all_params, bufsize=1, universal_newlines=True, stdout=stdout_file, stderr=subprocess.STDOUT)
            with GracefulInterruptHandler() as handler:
                returncode = proc.wait()
                interrupted = handler.interrupted

        task_status = None
        was_error = False
        if returncode < 0:
            print(colored("Process was killed by a signal!","magenta"))
            was_error = True
        elif skip_complete:
            print(colored("Skipping saving the result (skip_complete=True)"))
        else:
            task_status = TrainExitStatus(returncode)

            if task_status == TrainExitStatus.success:
                print(colored("SUCCESS! Reached desired correctness.","green"))
                with open(completed_file,'w') as f:
                    f.write("SUCCESS\n")
            elif task_status == TrainExitStatus.reached_update_limit:
                print(colored("FAIL! Reached update limit without attaining desired correctness.","red"))
                with open(completed_file,'w') as f:
                    f.write("FAIL_UPDATE_LIMIT\n")
            elif task_status == TrainExitStatus.overfitting:
                print(colored("FAIL! Detected overfitting.","red"))
                with open(completed_file,'w') as f:
                    f.write("FAIL_OVERFITTING\n")
            elif task_status in (TrainExitStatus.error, TrainExitStatus.malformed_command):
                print(colored("Got an error; skipping for now. See {} for details.".format(stdout_fn),"magenta"))
                was_error = True
            elif task_status == TrainExitStatus.nan_loss:
                print(colored("NaN loss detected; skipping for now.","magenta"))
                was_error = True
        
        if task_status == TrainExitStatus.interrupted or interrupted:
            print(colored("Process was interrupted! Stopping...","cyan"))
            break

        if was_error and stop_on_error:
            print(colored("Got an error. Exiting...","cyan"))
            break
