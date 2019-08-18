from Experiments import *
from BanditDriver import BanditDriver
import pickle
import argparse
import collections
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd

# from real_experiment_analysis import *
SingleResult = collections.namedtuple('SingleResult', 
    ['name', 'experiment', 'rewards', 'opt_rewards', 'opt_real_rewards', 
    'pulled_arms', 'opt_arms', 'opt_real_arms', 'regret'])
BigExperiment = collections.namedtuple('BigExperiment', ['ratio', 'context_size', 'c', 'error_mean', 'bandit', 'delta', 'T'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs Group Fair MAB experiments')
    parser.add_argument('type', help="type")
    parser.add_argument('dir', help='dir')
    args = parser.parse_args()

    if args.type == "remap":
        files = [f for f in os.listdir(args.dir) if os.path.isfile(os.path.join(args.dir, f))]
        remapped_experiments = {}
        for filename in files:
            # Get information from the filename
            print(filename)
            try:
                i1 = filename.index('_context_')
            except:
                continue
            ratio = filename[len('ratio_'):i1]
            i2 = filename.index('_c_')
            context_size = filename[i1 + len('_context_'):i2]
            i1 = i2
            i2 = filename.index('_error_')
            c = filename[i1 + len('_c_'):i2]
            i1 = i2
            i2 = filename.index('_run')
            error_mean = filename[i1+len('_error_'):i2]
            
            #Open file
            with open(os.path.join(args.dir, filename), 'rb') as f:
                results = pickle.load(f)
            # Rewrite experiments
            for experiment in results['experiment_results']:
                for i in range(len(experiment.experiments)):
                    big_experiment = BigExperiment(ratio, context_size, c, error_mean, 
                        experiment.experiments[i].bandit, experiment.experiments[i].delta, 
                        experiment.experiments[i].T)
                    name = "_".join([str(x) for x in [ratio, context_size, c, error_mean, 
                        experiment.experiments[i].bandit, experiment.experiments[i].delta, 
                        experiment.experiments[i].T]])
                    result = SingleResult(name, big_experiment, experiment.rewards[i], 
                        experiment.opt_rewards[i], experiment.opt_real_rewards[i],
                        experiment.pulled_arms[i], experiment.opt_arms[i], 
                        experiment.opt_real_arms[i], experiment.regret[i])
                    if name in remapped_experiments:
                        remapped_experiments[name].append(result)
                    else:
                        remapped_experiments[name] = [result]

        with open(os.path.join(args.dir, 'remapped_experiments.pkl'), 'wb') as f:
            pickle.dump(remapped_experiments, f)
