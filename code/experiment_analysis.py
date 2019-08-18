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
BigExperiment = collections.namedtuple('BigExperiment', ['ratio', 'context_size', 'c', 'error_mean', 'bandit', 'delta', 'T', 'arms', 'sensitive_group'])
AverageResult = collections.namedtuple('AverageResult', ['name', 'mean', 'std'])

algorithms = ["TopInterval", "IntervalChaining", "GroupFairTopInterval"]
colors = ['blue','red','green']


def plot_two_things(averages1, averages2, filename, title, ylabel, xlabel, a1_type, a2_type):
    # Sort by Time
    for key in averages1.keys():
        averages1[key].sort(key=lambda x: x.name)
    for key in averages2.keys():
        averages2[key].sort(key=lambda x: x.name)
    # <ake graphs
    fig, ax = plt.subplots()
    for i in range(len(algorithms)):
        algorithm = algorithms[i]
        Ts = [x.name for x in averages1[algorithm]]
        means = np.array([x.mean for x in averages1[algorithm]])
        stds = np.array([x.std for x in averages1[algorithm]])
        ax.plot(Ts, means, label=algorithm + " " + a1_type , color=colors[i])
    for i in range(len(algorithms)):
        algorithm = algorithms[i]
        Ts = [x.name for x in averages2[algorithm]]
        means = np.array([x.mean for x in averages2[algorithm]])
        stds = np.array([x.std for x in averages2[algorithm]])
        ax.plot(Ts, means, label=algorithm + " " + a2_type , color=colors[i], linestyle='dashed')

    ax.set(xlabel=xlabel, ylabel=ylabel,
           title=title)
    ax.legend()
    ax.grid()
    plt.tight_layout()
    fig.savefig(filename)
    # plt.show()
    plt.close()

def plot_things(averages, filename, title, ylabel, xlabel='T'):
    # Sort by Time
    for key in averages.keys():
        averages[key].sort(key=lambda x: x.name)
    # Make graphs
    fig, ax = plt.subplots()
    for algorithm in algorithms:
        Ts = [x.name for x in averages[algorithm]]
        means = np.array([x.mean for x in averages[algorithm]])
        stds = np.array([x.std for x in averages[algorithm]])
        ax.plot(Ts, means, label=algorithm)
        # ax.fill_between(Ts, means + stds, means - stds)

    ax.set(xlabel=xlabel, ylabel=ylabel,
           title=title)
    ax.legend()
    ax.grid()
    plt.tight_layout()
    fig.savefig(filename)
    # plt.show()
    plt.close()

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
                i1 = filename.index('_ratio_')
            except:
                continue
            context_size = int(filename[len('context_'):i1])
            i2 = filename.index('_c_')
            ratio = int(filename[i1 + len('_ratio_'):i2])
            i1 = i2
            i2 = filename.index('_delta_')
            c = int(filename[i1 + len('_c_'):i2])
            i1 = i2
            i2 = filename.index('_error_')
            delta = float(filename[i1+len('_delta_'):i2])
            i1 = i2
            i2 = filename.index('_arms_')
            error_mean = int(filename[i1+len('_error_'):i2])
            i1 = i2
            i2 = filename.index('_run_')
            arms = int(filename[i1+len('_arms_'):i2])
            
            #Open file
            with open(os.path.join(args.dir, filename), 'rb') as f:
                results = pickle.load(f)

            sensitive_group = results['sensitive_group']
            # print(sensitive_group)
            # Rewrite experiments
            for experiment in results['experiment_results']:
                for i in range(len(experiment.experiments)):
                    big_experiment = BigExperiment(ratio, context_size, c, error_mean, 
                        experiment.experiments[i].bandit, experiment.experiments[i].delta, 
                        experiment.experiments[i].T, arms, sensitive_group)
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
    if args.type == "T":
        with open(os.path.join(args.dir, 'remapped_experiments.pkl'), 'rb') as f:
            experiments = pickle.load(f)
        average_real_regrets = {}
        average_regrets = {}
        average_pull = {}
        for key in experiments.keys():
            regret = []
            real_regret = []
            sensitive_pulled = []
            for experiment in experiments[key]:
                T = experiment.experiment.T
                algorithm = experiment.experiment.bandit
                regret.append(np.mean(np.array(experiment.opt_rewards) - np.array(experiment.rewards)))
                real_regret.append(np.mean(np.array(experiment.opt_real_rewards) - np.array(experiment.rewards)))
                num_sensitive = 0
                for arm in experiment.pulled_arms:
                    if experiment.experiment.sensitive_group[arm]:
                        num_sensitive += 1
                sensitive_pulled.append((num_sensitive*1.0)/len(experiment.pulled_arms))
                # break
            regret = AverageResult(T,np.mean(regret), np.std(regret))
            real_regret = AverageResult(T,np.mean(real_regret), np.std(real_regret))
            sensitive_pulled = AverageResult(T, np.mean(sensitive_pulled), np.std(sensitive_pulled))
            if algorithm in average_regrets:
                average_regrets[algorithm].append(regret)
                average_real_regrets[algorithm].append(real_regret)
                average_pull[algorithm].append(sensitive_pulled)
            else:
                average_regrets[algorithm] = [regret]
                average_real_regrets[algorithm] = [real_regret]
                average_pull[algorithm] = [sensitive_pulled]
        path = args.dir + "/graphs/"
        if not os.path.exists(path):
            os.makedirs(path)
        plot_two_things(average_regrets, average_real_regrets, os.path.join(path,'regret.png'), 'Regret over budget T', 'Regret', 'T', '', 'real')
        plot_things(average_pull, os.path.join(path, 'pulls.png'), 'Sensitive arms pulled', 'Percent sensitive arms pulled', 'T')
        
