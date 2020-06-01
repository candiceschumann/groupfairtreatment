from Experiments import *
from BanditDriver import BanditDriver
import pickle
import argparse
import collections
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
# matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 20})
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os
import pandas as pd
import json

SingleResult = collections.namedtuple('SingleResult', 
    ['name', 'experiment', 'rewards', 'opt_real_rewards', 
    'pulled_arms', 'opt_real_arms', 'regret'])
AverageResult = collections.namedtuple('AverageResult', ['name', 'mean', 'std'])
max_bytes = 2**31 - 1

def plot_things(averages, path, title, ylabel, xlabel='T', config=None):
    # Sort by Time
    for key in averages.keys():
        averages[key].sort(key=lambda x: x.name)
    # Make graphs
    for delta in config['deltas']:
        fig, ax = plt.subplots()
        for algorithm in config['algorithms']:
            sub_name = algorithm + "_" + str(delta)
            if algorithm == "GroupFairProportional":
                algorithm = "NaiveFair"
            Ts = [x.name for x in averages[sub_name]]
            means = np.array([x.mean for x in averages[sub_name]])
            stds = np.array([x.std for x in averages[sub_name]])
            ax.plot(Ts, means, label=algorithm,linewidth=3.0)
            ax.fill_between(Ts, means-stds, means+stds, alpha=0.3)
            # ax.fill_between(Ts, means + stds, means - stds)
        
        ax.set(xlabel=xlabel, ylabel=ylabel,
               title=title)
        fontP = FontProperties()
        fontP.set_size('x-small')
        ax.legend(loc='upper left',prop=fontP)
        ax.grid()
        plt.tight_layout()
        fig.savefig(path + str(delta) + ".png")
        # plt.show()
        plt.close()

def regret(experiments, output_dir, config):
    path = output_dir + "/regret/"
    if not os.path.exists(path):
        os.makedirs(path)
    # Get average regret with std
    averages = {}
    for key in experiments.keys():
        regret = []
        for experiment in experiments[key]:
            regret.append(np.mean(experiment.regret))
            T = experiment.experiment.T
            sub_name = experiment.experiment.bandit + "_" + str(experiment.experiment.delta)
        average_regret = AverageResult(T,np.mean(regret),np.std(regret))
        if sub_name in averages:
            averages[sub_name].append(average_regret)
        else:
            averages[sub_name] = [average_regret]
    plot_things(averages, path, "Regret", "Average Regret", config=config)

def arms_pulled(experiments, sensitive_group, output_dir, config):
    path = output_dir + "/arms_pulled/"
    if not os.path.exists(path):
        os.makedirs(path)
    # Get average sensitive arms pulled with std
    averages = {}
    for key in experiments.keys():
        sensitive_pulled = []
        for experiment in experiments[key]:
            num_sensitive = 0
            for arm in experiment.pulled_arms:
                if not sensitive_group[arm]:
                    num_sensitive += 1
            sensitive_pulled.append(1-(num_sensitive*1.0)/len(experiment.pulled_arms))
            T = experiment.experiment.T
            sub_name = experiment.experiment.bandit + "_" + str(experiment.experiment.delta)
        average_sensitive_pulled = AverageResult(T,np.mean(sensitive_pulled),np.std(sensitive_pulled))
        if sub_name in averages:
            averages[sub_name].append(average_sensitive_pulled)
        else:
            averages[sub_name] = [average_sensitive_pulled]
    print(averages)

    plot_things(averages, path, "% of sensitive arms pulled", "% sensitive arms", config=config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs Group Fair MAB experiment analysis')
    parser.add_argument('type', help="type")
    parser.add_argument('output_dir', help='dir')
    parser.add_argument('config_file', help='config')
    parser.add_argument('--in_dirs', nargs='*')
    args = parser.parse_args()

    if not os.path.exists(os.path.dirname(args.output_dir)):
            os.makedirs(os.path.dirname(args.output_dir))

    # Get the config file
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    print(config)

    if (args.in_dirs):
        # Collect the data and remap into something usable
        remapped_experiments = {}
        for in_dir in args.in_dirs:
            for in_file in os.listdir(in_dir):
                if (in_file != "remapped_experiments.pkl" and in_file != "groups.pkl") and in_file[-4:] == ".pkl":
                    filename = os.path.join(in_dir,in_file)
                    print(filename) 
                    with open(filename, 'rb') as f:
                        experiment = pickle.load(f)
                    print(experiment['sensitive_group'])
                    with open(args.output_dir + "groups.pkl", 'wb') as f:
                        pickle.dump(experiment['sensitive_group'],f)
                        groups = experiment['sensitive_group']
                    for experiment_result in experiment['experiment_results']:
                        for i in range(len(experiment_result.experiments)):
                            name = (experiment_result.experiments[i].bandit + "_" +
                                str(experiment_result.experiments[i].delta) + "_" +
                                str(experiment_result.experiments[i].T))
                            single_result = SingleResult(name, experiment_result.experiments[i], 
                                experiment_result.rewards[i], experiment_result.opt_real_rewards[i], 
                                experiment_result.pulled_arms[i], experiment_result.opt_real_arms[i], 
                                experiment_result.regret[i])
                            if name in remapped_experiments:
                                remapped_experiments[name].append(single_result)
                            else:
                                remapped_experiments[name] = [single_result]
                else:
                    print("skipping: " + in_file)
        bytes_out = pickle.dumps(remapped_experiments)
        with open(args.output_dir + "remapped_experiments.pkl",'wb') as f:
            for idx in range(0, len(bytes_out), max_bytes):
                f.write(bytes_out[idx:idx+max_bytes])
    elif ("remapped_experiments.pkl" in os.listdir(args.output_dir)):
        # Get the usable data if it exists
        with open(args.output_dir + "remapped_experiments.pkl", "rb") as f:
            remapped_experiments = pickle.load(f)
        with open(args.output_dir + "groups.pkl", "rb") as f:
            groups = pickle.load(f)
    else:
        # I don't know what data I am supposed to use.
        print("I NEED DATA PLEASE")

    if (args.type == "regret"):
        regret(remapped_experiments, args.output_dir, config)
    elif (args.type == "arms_pulled"):
        arms_pulled(remapped_experiments, groups, args.output_dir, config)
    elif (args.type == "all"):
        regret(remapped_experiments, args.output_dir, config)
        arms_pulled(remapped_experiments, groups, args.output_dir, config)




