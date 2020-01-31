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
import os
import pandas as pd


# from real_experiment_analysis import *
SingleResult = collections.namedtuple('SingleResult', 
    ['name', 'experiment', 'rewards', 'real_rewards', 'opt_rewards', 'opt_real_rewards', 
    'pulled_arms', 'opt_arms', 'opt_real_arms', 'regret'])
BigExperiment = collections.namedtuple('BigExperiment', ['ratio', 'context_size', 'c', 'error_mean', 'bandit', 'delta', 'T', 'arms', 'sensitive_group'])
AverageResult = collections.namedtuple('AverageResult', ['name', 'mean', 'std'])

algorithms = ["TopInterval", "IntervalChaining", "GroupFairTopInterval", "MultiGroupFairTopInterval"]
colors = ['blue','red','green', 'orange']


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
        ax.plot(Ts, means, label=algorithm + " " + a1_type , color=colors[i], linewidth=3.0)
    for i in range(len(algorithms)):
        algorithm = algorithms[i]
        Ts = [x.name for x in averages2[algorithm]]
        means = np.array([x.mean for x in averages2[algorithm]])
        stds = np.array([x.std for x in averages2[algorithm]])
        ax.plot(Ts, means, label=algorithm + " " + a2_type , color=colors[i], linestyle='dashed', linewidth=3.0)

    ax.set(xlabel=xlabel, ylabel=ylabel,
           title=title)
    # ax.legend(facecolor="white", framealpha=1.0)
    ax.grid()
    plt.tight_layout()
    fig.savefig(filename)
    ax.legend(facecolor="white", framealpha=1.0)
    fig.savefig(filename+"test.png")
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
        ax.plot(Ts, means, label=algorithm, linewidth=3.0)
        # ax.fill_between(Ts, means + stds, means - stds)

    ax.set(xlabel=xlabel, ylabel=ylabel,
           title=title)
    # ax.legend(facecolor="white", framealpha=1.0)
    ax.grid()
    plt.tight_layout()
    fig.savefig(filename)
    ax.legend(facecolor="white", framealpha=1.0)
    fig.savefig(filename+"test.png")
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
                        experiment.real_rewards[i], 
                        experiment.opt_rewards[i], experiment.opt_real_rewards[i],
                        experiment.pulled_arms[i], experiment.opt_arms[i], 
                        experiment.opt_real_arms[i], experiment.regret[i])
                    if name in remapped_experiments:
                        remapped_experiments[name].append(result)
                    else:
                        remapped_experiments[name] = [result]

        with open(os.path.join(args.dir, 'remapped_experiments.pkl'), 'wb') as f:
            pickle.dump(remapped_experiments, f)
    else:
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
                if args.type == 'T':
                    x = experiment.experiment.T
                elif args.type == 'context_size':
                    x = experiment.experiment.context_size
                elif args.type == 'c':
                    x = experiment.experiment.c
                elif args.type == 'error':
                    x = experiment.experiment.error_mean
                elif args.type == 'delta':
                    x = experiment.experiment.delta
                elif args.type == 'arms':
                    x = experiment.experiment.arms
                elif args.type == 'ratio':
                    x = experiment.experiment.ratio
                algorithm = experiment.experiment.bandit
                regret.append(np.mean(np.array(experiment.opt_rewards) - np.array(experiment.rewards)))
                real_regret.append(np.mean(np.array(experiment.opt_real_rewards) - np.array(experiment.real_rewards)))
                num_sensitive = 0
                for arm in experiment.pulled_arms:
                    if experiment.experiment.sensitive_group[arm]:
                        num_sensitive += 1
                sensitive_pulled.append((num_sensitive*1.0)/len(experiment.pulled_arms))
                # break
            regret = AverageResult(x,np.mean(regret), np.std(regret))
            real_regret = AverageResult(x,np.mean(real_regret), np.std(real_regret))
            sensitive_pulled = AverageResult(x, np.mean(sensitive_pulled), np.std(sensitive_pulled))
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
        if args.type == 'T':
            x_label = 'T'
        elif args.type == 'context_size':
            x_label = 'context size d'
        elif args.type == 'c':
            x_label = 'c'
        elif args.type == 'error':
            x_label = 'error mean'
        elif args.type == 'delta':
            x_label = 'delta'
        elif args.type == 'arms':
            x_label = 'Number of arms'
        elif args.type == 'ratio':
            x_label = 'Number of sensitive arms'
        plot_two_things(average_regrets, average_real_regrets, os.path.join(path,args.type + '_regret.png'), 'Regret', 'Regret', x_label, '', 'real')
        plot_things(average_pull, os.path.join(path,args.type + '_pulls.png'), 'Sensitive arms pulled', 'Percent sensitive arms pulled', x_label)
    # if args.type == "T":
    #     with open(os.path.join(args.dir, 'remapped_experiments.pkl'), 'rb') as f:
    #         experiments = pickle.load(f)
    #     average_real_regrets = {}
    #     average_regrets = {}
    #     average_pull = {}
    #     for key in experiments.keys():
    #         regret = []
    #         real_regret = []
    #         sensitive_pulled = []
    #         for experiment in experiments[key]:
    #             T = experiment.experiment.T
    #             algorithm = experiment.experiment.bandit
    #             regret.append(np.mean(np.array(experiment.opt_rewards) - np.array(experiment.rewards)))
    #             real_regret.append(np.mean(np.array(experiment.opt_real_rewards) - np.array(experiment.rewards)))
    #             num_sensitive = 0
    #             for arm in experiment.pulled_arms:
    #                 if experiment.experiment.sensitive_group[arm]:
    #                     num_sensitive += 1
    #             sensitive_pulled.append((num_sensitive*1.0)/len(experiment.pulled_arms))
    #             # break
    #         regret = AverageResult(T,np.mean(regret), np.std(regret))
    #         real_regret = AverageResult(T,np.mean(real_regret), np.std(real_regret))
    #         sensitive_pulled = AverageResult(T, np.mean(sensitive_pulled), np.std(sensitive_pulled))
    #         if algorithm in average_regrets:
    #             average_regrets[algorithm].append(regret)
    #             average_real_regrets[algorithm].append(real_regret)
    #             average_pull[algorithm].append(sensitive_pulled)
    #         else:
    #             average_regrets[algorithm] = [regret]
    #             average_real_regrets[algorithm] = [real_regret]
    #             average_pull[algorithm] = [sensitive_pulled]
    #     path = args.dir + "/graphs/"
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     plot_two_things(average_regrets, average_real_regrets, os.path.join(path,'regret.png'), 'Regret over budget T', 'Regret', 'T', '', 'real')
    #     plot_things(average_pull, os.path.join(path, 'pulls.png'), 'Sensitive arms pulled', 'Percent sensitive arms pulled', 'T')
    # if args.type == "context_size":
    #     with open(os.path.join(args.dir, 'remapped_experiments.pkl'), 'rb') as f:
    #         experiments = pickle.load(f)
    #     average_real_regrets = {}
    #     average_regrets = {}
    #     average_pull = {}
    #     for key in experiments.keys():
    #         regret = []
    #         real_regret = []
    #         sensitive_pulled = []
    #         for experiment in experiments[key]:
    #             context_size = experiment.experiment.context_size
    #             algorithm = experiment.experiment.bandit
    #             regret.append(np.mean(np.array(experiment.opt_rewards) - np.array(experiment.rewards)))
    #             real_regret.append(np.mean(np.array(experiment.opt_real_rewards) - np.array(experiment.rewards)))
    #             num_sensitive = 0
    #             for arm in experiment.pulled_arms:
    #                 if experiment.experiment.sensitive_group[arm]:
    #                     num_sensitive += 1
    #             sensitive_pulled.append((num_sensitive*1.0)/len(experiment.pulled_arms))
    #             # break
    #         regret = AverageResult(context_size,np.mean(regret), np.std(regret))
    #         real_regret = AverageResult(context_size,np.mean(real_regret), np.std(real_regret))
    #         sensitive_pulled = AverageResult(context_size, np.mean(sensitive_pulled), np.std(sensitive_pulled))
    #         if algorithm in average_regrets:
    #             average_regrets[algorithm].append(regret)
    #             average_real_regrets[algorithm].append(real_regret)
    #             average_pull[algorithm].append(sensitive_pulled)
    #         else:
    #             average_regrets[algorithm] = [regret]
    #             average_real_regrets[algorithm] = [real_regret]
    #             average_pull[algorithm] = [sensitive_pulled]
    #     path = args.dir + "/graphs/"
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     plot_two_things(average_regrets, average_real_regrets, os.path.join(path,'regret.png'), 'Regret over context size', 'Regret', 'context size d', '', 'real')
    #     plot_things(average_pull, os.path.join(path, 'pulls.png'), 'Sensitive arms pulled', 'Percent sensitive arms pulled', 'context size d')
    # if args.type == "c":
    #     with open(os.path.join(args.dir, 'remapped_experiments.pkl'), 'rb') as f:
    #         experiments = pickle.load(f)
    #     average_real_regrets = {}
    #     average_regrets = {}
    #     average_pull = {}
    #     for key in experiments.keys():
    #         regret = []
    #         real_regret = []
    #         sensitive_pulled = []
    #         for experiment in experiments[key]:
    #             c = experiment.experiment.c
    #             algorithm = experiment.experiment.bandit
    #             regret.append(np.mean(np.array(experiment.opt_rewards) - np.array(experiment.rewards)))
    #             real_regret.append(np.mean(np.array(experiment.opt_real_rewards) - np.array(experiment.rewards)))
    #             num_sensitive = 0
    #             for arm in experiment.pulled_arms:
    #                 if experiment.experiment.sensitive_group[arm]:
    #                     num_sensitive += 1
    #             sensitive_pulled.append((num_sensitive*1.0)/len(experiment.pulled_arms))
    #             # break
    #         regret = AverageResult(c,np.mean(regret), np.std(regret))
    #         real_regret = AverageResult(c,np.mean(real_regret), np.std(real_regret))
    #         sensitive_pulled = AverageResult(c, np.mean(sensitive_pulled), np.std(sensitive_pulled))
    #         if algorithm in average_regrets:
    #             average_regrets[algorithm].append(regret)
    #             average_real_regrets[algorithm].append(real_regret)
    #             average_pull[algorithm].append(sensitive_pulled)
    #         else:
    #             average_regrets[algorithm] = [regret]
    #             average_real_regrets[algorithm] = [real_regret]
    #             average_pull[algorithm] = [sensitive_pulled]
    #     path = args.dir + "/graphs/"
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     plot_two_things(average_regrets, average_real_regrets, os.path.join(path,'regret.png'), 'Regret', 'Regret', 'c', '', 'real')
    #     plot_things(average_pull, os.path.join(path, 'pulls.png'), 'Sensitive arms pulled', 'Percent sensitive arms pulled', 'c')
    # if args.type == "error":
    #     with open(os.path.join(args.dir, 'remapped_experiments.pkl'), 'rb') as f:
    #         experiments = pickle.load(f)
    #     average_real_regrets = {}
    #     average_regrets = {}
    #     average_pull = {}
    #     for key in experiments.keys():
    #         regret = []
    #         real_regret = []
    #         sensitive_pulled = []
    #         for experiment in experiments[key]:
    #             error_mean = experiment.experiment.error_mean
    #             algorithm = experiment.experiment.bandit
    #             regret.append(np.mean(np.array(experiment.opt_rewards) - np.array(experiment.rewards)))
    #             real_regret.append(np.mean(np.array(experiment.opt_real_rewards) - np.array(experiment.rewards)))
    #             num_sensitive = 0
    #             for arm in experiment.pulled_arms:
    #                 if experiment.experiment.sensitive_group[arm]:
    #                     num_sensitive += 1
    #             sensitive_pulled.append((num_sensitive*1.0)/len(experiment.pulled_arms))
    #             # break
    #         regret = AverageResult(error_mean,np.mean(regret), np.std(regret))
    #         real_regret = AverageResult(error_mean,np.mean(real_regret), np.std(real_regret))
    #         sensitive_pulled = AverageResult(error_mean, np.mean(sensitive_pulled), np.std(sensitive_pulled))
    #         if algorithm in average_regrets:
    #             average_regrets[algorithm].append(regret)
    #             average_real_regrets[algorithm].append(real_regret)
    #             average_pull[algorithm].append(sensitive_pulled)
    #         else:
    #             average_regrets[algorithm] = [regret]
    #             average_real_regrets[algorithm] = [real_regret]
    #             average_pull[algorithm] = [sensitive_pulled]
    #     path = args.dir + "/graphs/"
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     plot_two_things(average_regrets, average_real_regrets, os.path.join(path,'regret.png'), 'Regret', 'Regret', 'error mean', '', 'real')
    #     plot_things(average_pull, os.path.join(path, 'pulls.png'), 'Sensitive arms pulled', 'Percent sensitive arms pulled', 'error mean')

