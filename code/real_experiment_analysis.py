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

SingleResult = collections.namedtuple('SingleResult', 
    ['name', 'experiment', 'rewards', 'opt_real_rewards', 
    'pulled_arms', 'opt_real_arms', 'regret'])
AverageResult = collections.namedtuple('AverageResult', ['name', 'mean', 'std'])

# algorithms = ["TopInterval", "IntervalChaining", "Random", "GroupFairTopInterval"]
algorithms = ["TopInterval", "IntervalChaining", "GroupFairTopInterval"]
deltas = [.1, .2, .3, .4, .5]

def plot_things(averages, path, title, ylabel, xlabel='T'):
    # Sort by Time
        for key in averages.keys():
            averages[key].sort(key=lambda x: x.name)
        # Make graphs
        for delta in deltas:
            fig, ax = plt.subplots()
            for algorithm in algorithms:
                sub_name = algorithm + "_" + str(delta)
                Ts = [x.name for x in averages[sub_name]]
                means = np.array([x.mean for x in averages[sub_name]])
                stds = np.array([x.std for x in averages[sub_name]])
                ax.plot(Ts, means, label=algorithm,linewidth=3.0)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs Group Fair MAB experiments')
    parser.add_argument('type', help="type")
    parser.add_argument('filename', help="Filename")
    parser.add_argument('dir', help='dir')
    args = parser.parse_args()



    if args.type == 'remap':
        with open(args.filename, 'rb') as f:
            experiment = pickle.load(f)
        # print(experiment['experiment_results'][0])
        remapped_experiments = {}
        for experiment_result in experiment['experiment_results']:
            # print(experiment_result.experiments)
            for i in range(len(experiment_result.experiments)):
                # print(experiment_result.experiments[i])
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

        print(remapped_experiments.keys())
        with open(args.dir + "remapped_experiments.pkl",'wb') as f:
            pickle.dump(remapped_experiments,f)
        print(experiment['sensitive_group'])
        with open(args.dir + "groups.pkl", 'wb') as f:
            pickle.dump(experiment['sensitive_group'],f)
    elif args.type == "regret":
        with open(args.filename, 'rb') as f:
            experiments = pickle.load(f)
        path = args.dir + "/regret/"
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
        plot_things(averages, path, "Regret", "Average Regret")
            # break
    elif args.type == "arms_pulled":
        with open(args.dir + '/groups.pkl', 'rb') as f:
            sensitive_group = pickle.load(f)
        with open(args.filename, 'rb') as f:
            experiments = pickle.load(f)
        path = args.dir + "/arms_pulled/"
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
                sensitive_pulled.append((num_sensitive*1.0)/len(experiment.pulled_arms))
                T = experiment.experiment.T
                sub_name = experiment.experiment.bandit + "_" + str(experiment.experiment.delta)
            average_sensitive_pulled = AverageResult(T,np.mean(sensitive_pulled),np.std(sensitive_pulled))
            if sub_name in averages:
                averages[sub_name].append(average_sensitive_pulled)
            else:
                averages[sub_name] = [average_sensitive_pulled]
        print(averages)

        plot_things(averages, path, "% of sensitive arms pulled", "% sensitive arms")



