# Graphical State Transitions Framework

This is the code supporting my paper ["Learning Graphical State Transitions"][gpaper], published in ICLR 2017. It consists of

- implementation of each of the graph transformations described in the paper (in the transformation_modules subdirectory)
- implementation of the Gated Graph Transformer Neural Network model (model.py)
- a tool to convert a folder of tasks written in a textual form with JSON graphs into a series of python pickle files with appropriate metadata (ggtnn_graph_parse.py)
- a harness to train multiple tasks in sequence (run_harness.py)
- a helper executable to train on the sequence of bAbI tasks (do_babi_run.py)
- a set of task-generators to generate data for particular tasks, such as the Turing machine and automaton tasks discussed in the paper (in the task_generators subdirectory)
- a tool to enable visualization of the produced graphs, either interactively in [Jupyter][] (by importing display_graph.py and calling `graph_display`) or noninteractively using [phantomjs][] (by running display_graph.js) 
- other miscellaneous utilities

Note that there were also modifications to the bAbI task generation code in order to extend them with graph information. For those, see [this repository][bAbi-mine].

To use the model, you will need [python 3.5][] or later with [Theano][] installed. If you wish to visualize the results, you will also need [matplotlib][] and [Jupyter][], and will need [phantomjs][] to generate images noninteractively. Additionally, you will need to have `floatX=float32` in your `.theanorc` file to compile models correctly.

[gpaper]: http://openreview.net/pdf?id=HJ0NvFzxl
[babi-mine]: https://github.com/hexahedria/bAbI-tasks
[python 3.5]: https://www.python.org/
[Theano]: http://deeplearning.net/software/theano/
[matplotlib]: http://matplotlib.org/
[Jupyter]: http://jupyter.org/
[phantomjs]: http://phantomjs.org/

## Quick start
This section is intended to be a brief guide to reproducing some of the results from the paper. For more details, see the following sections, or the code itself.

The first step is to create the actual training files. This can be done by cloning the [modified bAbI tasks repo][babi-mine], and then running the script `generate_graph_tasks.sh`, which will produce a large number of graphs in the `output` directory of the repo.

The next step is to train the model using the `do_babi_run.py` helper script. For example, you might run
```python
python3 do_babi_run.py path/to/bAbI-tasks/output ./model_results
```
which will train the model on all of the bAbI tasks with default parameters, and save the results into `./model_results`.

Arguments accepted by `do_babi_run.py`:

- `--including-only TASK_ID_1 TASK_ID_2 ...` will cause it to only train the model on the specific tasks given (where each TASK_ID represents the numerical index of the desired task).
- `--dataset-sizes SIZE_1 SIZE_2 ...` will cause it to only train the model with the specified sizes of dataset. To train only with the full dataset, pass `--dataset-sizes 1000`. The default is equivalent to `--dataset-sizes 50 100 250 500 1000`.
- `--direct-reference` and `--no-direct-reference` will cause it to only train with or without direct reference, respectively. By default, it will train both types of model; this forces it to only train one.

The `do_babi_run.py` script sets specific parameters based on each task. In particular, it uses the appropriate output format for each network, and also enables or disables the intermediate propagation step depending on the complexity of the task. For each task, it then sets up multiple training runs with differently sized subsets of the input dataset, and also configures versions of the model with direct reference enabled and disabled. Internally, it then defers to the `run_harness.py` module, which runs all of those tasks in sequence.

Additional arguments passed to `do_babi_run.py` are forwarded unchanged to the `main.py` script, which are described below. Note that the `do_babi_run.py` script automatically sets many of these arguments, so be careful to avoid conflicts.

## Non-bAbI graphs
The model also has support for tasks that are not from the bAbI dataset. However, the training process is somewhat more complex.

First, the graphs have to be obtained in the correct textual format. The parser script expects to see a file that is divided into multiple stories, where each story represents one training example and all stories are independent. A story consists of a series of sequentially numbered lines, which are either statements or queries, and end with a final query.

A *statement* should be of the form
```
<number> <words>={"nodes":[node1name,node2name,...], "edges":[{ "type":e1type,"from":e1sourcename,"to":e1dest}, { "type":e2type,"from":e2sourcename,"to":e2dest},...]}
```
where the names, types, sources, and destinations are all strings, and the sources and destinations match nodes in the node list. Note that each node name must be distinct. During processing, each of the words, node names, and edge types seen in the dataset will be mapped to unique integer indices. If your graph should have multiple nodes of the same type, the nodes can be disambiguated using a "#" and a suffix. For instance "cell#0" and "cell#1" will both be mapped to the "cell" node type. The graph should represent the desired state of the network after processing this statement.

A *query* should be of the form
```
<number> <words> <tab character> <answer> <optional tab character> <ignored>
```
The query is given by the words before the first tab character, and the network answer is given by the word or words after. Content after the second tab character is ignored, but is allowed for compatibility with the bAbI task format. (If the task has no meaningful query, then a simple empty string should be used for both the words and the answer.)

In order for direct reference to work correctly, the name of the node in the graph should be the same as the word that refers to that node in the statement or query. So a node that will be addressed by "Mary" in the statements and queries should be called "Mary" (or "Mary#suffix") in the graph node list. If you do not want to use direct reference, the names of the graph elements are arbitrary.

As an example, this file (excerpted) is taken from the bAbI graph dataset:
```
1 Mary journeyed to the bathroom.={"nodes":["Mary","bathroom"],"edges":[{ "type":"actor_is_in_location","from":"Mary","to":"bathroom"}]}
2 Mary moved to the hallway.={"nodes":["Mary","hallway"],"edges":[{ "type":"actor_is_in_location","from":"Mary","to":"hallway"}]}
3 Where is Mary?    hallway 2
4 Sandra travelled to the hallway.={"nodes":["Mary","hallway","Sandra"],"edges":[{ "type":"actor_is_in_location","from":"Mary","to":"hallway"},{ "type":"actor_is_in_location","from":"Sandra","to":"hallway"}]}
5 Daniel travelled to the office.={"nodes":["Mary","Daniel","Sandra","hallway","office"],"edges":[{ "type":"actor_is_in_location","from":"Daniel","to":"office"},{ "type":"actor_is_in_location","from":"Mary","to":"hallway"},{ "type":"actor_is_in_location","from":"Sandra","to":"hallway"}]}
6 Where is Daniel?  office  5
1 Mary travelled to the bathroom.={"nodes":["Mary","bathroom"],"edges":[{ "type":"actor_is_in_location","from":"Mary","to":"bathroom"}]}
2 Mary travelled to the garden.={"nodes":["Mary","garden"],"edges":[{ "type":"actor_is_in_location","from":"Mary","to":"garden"}]}
(etc)
```

See the `task_generators` subdirectory for a collection of scripts that generate output in this format, including the Turing machine and automaton generators.

After obtaining a correctly-formatted story file, that file must be parsed and preprocessed to allow training. You can parse the file by running
```
python3 ggtnn_graph_parse.py path_to_file
```
which will create a directory and populate it with preprocessed training data. 

Note that in the process, it scans the file and uses it to determine the mapping of words to indices that will be used by the network, as well as the maximum lengths of various components of the model, which it stores in a metadata file. Models trained with one metadata file may not be able to correctly run on examples with a different metadata file. If you would prefer the network to not recompute the metadata and instead use an existing metadata file (for example to ensure that the training and testing sets both use the same metadata) you can pass an existing metadata file with the `--metadata-file` argument. You can also view a metadata file using the `metadata-display.py` helper script.

Finally, you can actually train the model on the dataset. You will need to pass a large number of parameters to completely configure the model for your task. For example, to train the model on the automaton task, you might run

```
python3 main.py task_output/automaton_30_train category 20 --outputdir output_auto30 --validation task_output/automaton_30_valid --mutable-nodes --dynamic-nodes --propagate-intermediate --direct-reference --num-updates 100000 --batch-size 100 --batch-adjust 28000000 --resume-auto --no-query
```

In the next section, I will describe some of the parameters of `main.py` that can be set, and what their uses are.

## Overview of parameters for main.py

The `main.py` script is the actual entry point for training a model. It has a large number of parameters, which are used to configure the model and determine what to do. For a full overview of all of the options available, you can run `python3 main.py -h`. In this section I will summarize the most important commands and their usage.

The simplest form of the invocation is
```
python3 main.py task_dir output_form state_width
```
where

- `task_dir` is the path to the directory created by the preprocessing step
- `output_form` gives the form of the output: it should be `category` if every answer is a single word; `set` if answers have multiple words, each word can appear at most once, and order does not matter; and `sequence` if answers can have multiple words but order does matter and there could be repeats.
- `state_width` determines the size of the state vector at each node. Since every node has a different state vector, making this parameter large can make it easier to overfit, but also allows more complex processing.

In addition to these, you can also pass parameters to configure other aspects of the model and run process.

### Model parameters

These parameters determine how the model actually is configured and run. 

```
  --process-repr-size PROCESS_REPR_SIZE
                        Width of intermediate representations (default: 50)
  --mutable-nodes       Make nodes mutable (default: False)
  --wipe-node-state     Wipe node state before the query (default: False)
  --direct-reference    Use direct reference for input, based on node names
                        (default: False)
  --dynamic-nodes       Create nodes after each sentence. (Otherwise, create
                        unique nodes at the beginning) (default: False)
  --propagate-intermediate
                        Run a propagation step after each sentence (default:
                        False)
  --no-graph            Don't train using graph supervision
  --no-query            Don't train using query supervision
```

Although not given by default, you will likely want to use `--mutable-nodes` and `--dynamic-nodes` for tasks with any complex processing involved; this creates the equivalent of the GGT-NN model in the paper. Otherwise, nodes will not be created at each step, and existing nodes will not update their states. You may also want to want to use `--direct-reference`, as it tends to increase performance. The `--propagate-intermediate` argument should be used if nodes need to exchange information in order to update their intermediate states correctly (for example, if the placement of new nodes depends on edges between other nodes). The `--no-query` argument can be passed if the task does not have a meaningful query and will disable the query processing in the model.

### Training parameters

These parameters affect the model training process. Most should be self explanatory.

```
  --num-updates NUM_UPDATES
                        How many iterations to train (default: 10000)
  --batch-size BATCH_SIZE
                        Batch size to use (default: 10)
  --learning-rate LEARNING_RATE
                        Use this learning rate (default: None)
  --dropout-keep DROPOUT_KEEP
                        Use dropout, with this keep chance (default: 1)
  --restrict-dataset NUM_STORIES
                        Restrict size of dataset to this (default: None)
  --validation VALIDATION_DIR
                        Parsed directory of validation tasks (default: None)
  --validation-interval VALIDATION_INTERVAL
                        Check validation after this many iterations (default:
                        1000)
  --stop-at-accuracy STOP_AT_ACCURACY
                        Stop training once it reaches this accuracy on
                        validation set (default: None)
  --stop-at-loss STOP_AT_LOSS
                        Stop training once it reaches this loss on validation
                        set (default: None)
  --stop-at-overfitting STOP_AT_OVERFITTING
                        Stop training once validation loss is this many times
                        higher than train loss (default: None)
  --batch-adjust BATCH_ADJUST
                        If set, ensure that size of edge matrix does not
                        exceed this (default: None)
```

The `--batch-adjust` argument can be used to prevent out-of-memory errors for large datasets. It uses a heuristic based on the size of the edge matrix to try to adjust the size of the batch based on the length of the input data. Good values of this should be determined by trial and error (with the bAbI I found a value of about 28000000 to work on my machine).

### IO Parameters

These parameters configure how the script performs I/O operations.
```
  --outputdir OUTPUTDIR
                        Directory to save output in (default: output)
  --save-params-interval TRAIN_SAVE_PARAMS
                        Save parameters after this many iterations (default:
                        1000)
  --final-params-only   Don't save parameters while training, only at the end.
                        (default: None)
  --set-exit-status     Give info about training status in the exit status
                        (default: False)
  --autopickle PICKLEDIR
                        Automatically cache model in this directory (default:
                        None)
  --pickle-model MODELFILE
                        Save the compiled model to a file (default: None)
  --unpickle-model MODELFILE
                        Load the model from a file instead of compiling it
                        from scratch (default: None)
  --interrupt-file INTERRUPT_FILE
                        Interrupt training if this file appears (default:
                        None)
  --resume TIMESTEP PARAMFILE
                        Where to restore from: timestep, and file to load
                        (default: None)
  --resume-auto         Automatically restore from a previous run using output
                        directory (default: False)
```

To speed up repeated uses of the model, I recommend using the `--autopickle` argument with a particular model-cache directory. The script will automatically determine a unique name for each model version and assign it to a given hash value, and then will try to load a cached model based on this hash. If it fails to find one, it will compile the model as normal and then save it into the directory based on the hash.

Additionally, if the training process is interrupted, the `--resume-auto` parameter will allow the training process to pick up where it left off. Otherwise, it will start over from iteration 0. You can also explicitly set a starting time using `--resume TIMESTEP PARAMFILE`.

### Alternate execution modes

The `main.py` script can also do other things in addition to training a model.
```
  --check-nan           Check for NaN. Slows execution (default: None)
  --check-debug         Debug mode. Slows execution (default: None)
  --just-compile        Don't run the model, just compile it (default: False)
  --visualize [BUCKET,STORY]
                        Visualise current state instead of training. Optional
                        parameter selects a particular story to visualize, and
                        should be of the form bucketnum,index (default: False)
  --visualize-snap      In visualization mode, snap to best option at each
                        timestep (default: False)
  --visualization-test  Like visualize, but use the correct graph instead of
                        the model's graph (default: False)
  --evaluate-accuracy   Evaluate accuracy of model (default: False)
```

The first two arguments are useful only if you are experiencing either NaN issues or an unexpected Theano error. The `--just-compile` is useful in conjunction with `--autopickle` in that it compiles and saves a model for later training.

The `--visualize` family of commands run the model on the input and generate visualization files, which can be converted into a diagram. If `--visualize` is used alone, the model will produce nodes whose strengths vary according to the strengths output by the model, producing "fuzzy" partial nodes. If `--visualize-snap` is also passed, the most likely option at each timestep will be selected instead, and the model will be forced to choose its actions with full strength.

The `--visualization-test` option is of limited use, and simply produces the visualization files correspoding to the correct graph structure from the dataset, but with the states from the model. (If you simply wish to visualize the correct graph structure, it is easier to use the `convert_story.py` script, which takes a story file and produces the graph visualization files.)

The `--evaluate-accuracy` argument evaluates the accuracy of the model over the dataset. In this mode, as in `--visualize-snap`, the most likely option at each timestep will be selected, and the model will be forced to choose its actions with full strength. If the result of the output exactly matches the correct result in the dataset, that sample is marked as a success, and otherwise it is a failure. It then prints out the fraction of samples that were successes. (When using this, pass the test dataset as the `task_dir` parameter.)

## Visualizing the results

After generating the visualization files, there are two ways to visualize them.

### Interactive visualization
To interactively visualize the output, you need to use Jupyter. In Jupyter, run
```
import numpy as np
from display.display_graph import graph_display, setup_graph_display
setup_graph_display()
def do_vis(direc, correct_graph=False, options={}):
    global results
    results = [np.load("{}/result_{}.npy".format(direc,i)) for i in (range(4) if correct_graph else range(1,5))]
    return graph_display(results,options)
```
Then to visualize a particular output, you can run `do_vis("path/to/vis/files")`. The `correct_graph` flag should be true if you are visualizing something that came from `convert_story.py`, and false if you are visualizing a network output. `options` should be a dictionary that contains various options for the visualization:

- `width` sets the width of the visualization
- `height` sets the height of the visualization
- `jitter` determines if any jitter is applied to the nodes
- `timestep` determines what timestep to start on (this is configurable interactively)
- `linkDistance` determines how long edges are (this is configurable interactively)
- `linkStrength` determines how much edges resist changes in length (this is configurable interactively)
- `gravity` determines how much nodes are attracted to one another (this is configurable interactively)
- `charge` determines how much nodes repel each other up close (this is configurable interactively)
- `extra_snap_specs` is a list of items of the form `{"id":0,"axis":"y","value": 40,"strength":0.6}`, where `id` specifies an individual node type, `axis` is "x" or "y", `value` gives a position, and `strength` determines how strongly the nodes of that type are attracted to that position. This can be used to impose constraints on the visualization based on the task (so that automaton cells line up in the middle in between values, for example). 
- `edge_strength_adjust` is a list of numbers, one for each edge type, which specify how strongly each type of edge resists being stretched.
- `noninteractive` will make the graph non-interactive, so that it doesn't animate or respond to user input.
    + `fullAlphaTicks` will determine how long to simulate forces in the graph before halting it, if running noninteractively.

If you install phantomjs, you can also directly export visualizations as images. First, make a file called `options.py` in the same directory as the visualization files, of the form
```
options = { ... }
```
where the dictionary has the options specified above. Then run
```
phantomjs display/generate_images.js path/to/vis/directory 1
```
where the trailing number can be increased to scale the size of the image produced.

## Training the extended sequential model (from the appendix)

The extended sequential model can be enabled using a few extra parameters. To train using `main.py` directly, use this argument:

```
  --sequence-aggregate-repr
                        Compute the query aggregate representation from the
                        sequence of graphs instead of just the last one
                        (default: False)
```

If you want to run the two bAbI tasks used in the paper with the extended model, you can pass `--run-sequential-set` to `do_babi_run.py`, which will have the same effect.

Note that in order to generate the dataset for this model, the additional history nodes do not need to be added. Thus in the files `WhereWasObject.lua` and `WhoWhatGave.lua`, the line including "augment_with_value_histories" should be commented out before generating the dataset for those tasks.