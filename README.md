# Group Fairness in Bandits with Biased Feedback

All code used in the above paper can be found here.

## Code description
Algorithms are broken down in to separate files in the algorithms folder. They contain information about how to perform an individual timestep. Each algorithm is run using the BanditDriver which asks the algorithm which arm to pull and then pulls the corresponding arm. Each arm is an instance of a ContextualArm (extensions to BiasContextualArm, ErrorContextualArm, and RealContextualArm).

An Experiment is run via the Experiment class. When an experiment is run all data from that run is saved in a pickle file (Note! This can take a lot of space as everything is saved).

## Architecture used
### Running Experiments
All experiments were run on a virtual machine with a 16 core Intel Xeon E312xx (Sandy Bridge) processor and 32GB of RAM.
### Running Analysis and Creating Graphs
All analysis and graph creation was done on a MacBook Pro with 2.5 GHz Intel Core i7 processor and 16 GB 1600 MHz DDR3 RAM

## Synthetic Experiments
To run the synthetic experiments found in the paper run the following:
```bash
./run_synth.sh
```
This runs a large amount of synthetic experiments sequentially. This can take ~2 days to run all of the experiments. A single experiment with a single run through of a single algorithm takes ~30 seconds depending on the hyperparameters used.

Once the experiments are complete, run the following for analysis and graph creation (Make sure to enter the appropriate path to data before running):
```bash
./experiment_analysis.sh
```
This collects all of the experiments into meaningful groups and creates graphs found in the paper.
## Real World Experiments
Before running real world experiments make sure to download both the [family income dataset](https://www.kaggle.com/grosvenpaul/family-income-and-expenditure) and the [COMPAS dataset](https://www.kaggle.com/danofer/compass).
Both datasets were filtered to exclude rows with missing context.

The real world datasets are run using a config file. The config files we used can be found in the configs folder. These files include the hyperparameters used. Please be sure to update the data file location in the config

To run an experiment given the config file run the following:
```bash
python3 run_experiment_config_file.py <path_to_config_file>
```

To run the analysis once that experiment is complete, run the following:
```bash
python3 experiment_config_analysis.py all <path_to_output_dir> <path_to_config_file> --in_dirs <path_to_experiment_dir>
```
Note that if multiple experiments are run multiple directories can be included in the --in_dirs.