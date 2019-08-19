from Experiments import Experiment
import os
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs Group Fair MAB experiments')
    parser.add_argument('exp', help="Experiment name")
    parser.add_argument('run_name', help="Run name (like number)")
    args = parser.parse_args()

    algorithms = ["TopInterval", "IntervalChaining", "GroupFairTopInterval"]
    runs = 10
    seeds = [np.random.random_integers(100000000) for _ in range(runs)]

    if args.exp == 'debug':
        arms = 10
        Ts = [100]
        context_size = 2
        ratio = 5
        groups = {"g1": [i for i in range(0, ratio)], "g2": [i for i in range(ratio, arms)]}
        sensitive_group = {}
        for i in range(0,ratio):
            sensitive_group[i] = True
        for i in range(ratio,arms):
            sensitive_group[i] = False
        c = 10
        hardness = {"g1": (0, c), "g2": (0, c)}
        delta = 0.1
        error_mean = 10
        error_std = 1
        filename = "../experiments/%s/context_%s_ratio_%s_c_%s_delta_%s_error_%s_arms_%s_run_%s" % (args.exp, 
            context_size, ratio, c, delta, error_mean, arms, args.run_name)
        if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
        experiment = Experiment(arms,context_size,groups,
                        bandit_types=[args.run_name],deltas=[delta],
                        Ts=Ts,arm_type="uniform",
                        filename=filename,
                        cs=hardness,sensitive_group=sensitive_group,group_mean=error_mean, 
                        group_std=error_std)
        experiment.run_x_experiments(1, seeds=[102])
    if args.exp == 'T':
        arms = 10
        Ts = [n * 10 for n in range(2, 101, 2)]
        context_size = 2
        ratio = 5
        groups = {"g1": [i for i in range(0, ratio)], "g2": [i for i in range(ratio, arms)]}
        sensitive_group = {}
        for i in range(0,ratio):
            sensitive_group[i] = True
        for i in range(ratio,arms):
            sensitive_group[i] = False
        c = 10
        hardness = {"g1": (0, c), "g2": (0, c)}
        delta = 0.1
        error_mean = 10
        error_std = 1
        filename = "../experiments/%s/context_%s_ratio_%s_c_%s_delta_%s_error_%s_arms_%s_run_%s" % (args.exp, 
            context_size, ratio, c, delta, error_mean, arms, args.run_name)
        if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
        experiment = Experiment(arms,context_size,groups,
                        bandit_types=algorithms,deltas=[delta],
                        Ts=Ts,arm_type="uniform",
                        filename=filename,
                        cs=hardness,sensitive_group=sensitive_group,group_mean=error_mean, 
                        group_std=error_std)
        experiment.run_x_experiments(runs)
    if args.exp == "context_size":
        arms = 10
        Ts = [1000]
        context_sizes = [n for n in range(2, 41, 2)]
        ratio = 5
        groups = {"g1": [i for i in range(0, ratio)], "g2": [i for i in range(ratio, arms)]}
        sensitive_group = {}
        for i in range(0,ratio):
            sensitive_group[i] = True
        for i in range(ratio,arms):
            sensitive_group[i] = False
        c = 10
        hardness = {"g1": (0, c), "g2": (0, c)}
        delta = 0.1
        error_mean = 10
        error_std = 1
        for context_size in context_sizes:
            filename = "../experiments/%s/context_%s_ratio_%s_c_%s_delta_%s_error_%s_arms_%s_run_%s" % (args.exp, 
                        context_size, ratio, c, delta, error_mean, arms, args.run_name)
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            experiment = Experiment(arms,context_size,groups,
                        bandit_types=algorithms,deltas=[delta],
                        Ts=Ts,arm_type="uniform",
                        filename=filename,
                        cs=hardness,sensitive_group=sensitive_group,group_mean=error_mean, 
                        group_std=error_std)
            experiment.run_x_experiments(runs,seeds=seeds)
    if args.exp == "c":
        arms = 10
        Ts = [1000]
        context_size = 2
        ratio = 5
        groups = {"g1": [i for i in range(0, ratio)], "g2": [i for i in range(ratio, arms)]}
        sensitive_group = {}
        for i in range(0,ratio):
            sensitive_group[i] = True
        for i in range(ratio,arms):
            sensitive_group[i] = False
        cs = [i for i in range(2,11)]
        delta = 0.1
        error_mean = 10
        error_std = 1
        for c in cs:
            hardness = {"g1": (0, c), "g2": (0, c)}
            filename = "../experiments/%s/context_%s_ratio_%s_c_%s_delta_%s_error_%s_arms_%s_run_%s" % (args.exp, 
                        context_size, ratio, c, delta, error_mean, arms, args.run_name)
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))

            experiment = Experiment(arms,context_size,groups,
                        bandit_types=algorithms,deltas=[delta],
                        Ts=Ts,arm_type="uniform",
                        filename=filename,
                        cs=hardness,sensitive_group=sensitive_group,group_mean=error_mean, 
                        group_std=error_std)
            experiment.run_x_experiments(runs,seeds=seeds)
    if args.exp == "error":
        arms = 10
        Ts = [1000]
        context_size = 2
        ratio = 5
        groups = {"g1": [i for i in range(0, ratio)], "g2": [i for i in range(ratio, arms)]}
        sensitive_group = {}
        for i in range(0,ratio):
            sensitive_group[i] = True
        for i in range(ratio,arms):
            sensitive_group[i] = False
        c = 10
        hardness = {"g1": (0, c), "g2": (0, c)}
        delta = 0.1
        error_means = [i for i in range(10)]
        error_std = 1
        for error_mean in error_means:
            filename = "../experiments/%s/context_%s_ratio_%s_c_%s_delta_%s_error_%s_arms_%s_run_%s" % (args.exp, 
                        context_size, ratio, c, delta, error_mean, arms, args.run_name)
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))

            experiment = Experiment(arms,context_size,groups,
                        bandit_types=algorithms,deltas=[delta],
                        Ts=Ts,arm_type="uniform",
                        filename=filename,
                        cs=hardness,sensitive_group=sensitive_group,group_mean=error_mean, 
                        group_std=error_std)
            experiment.run_x_experiments(runs,seeds=seeds)

    if args.exp == "ratio":
        arms = 10
        Ts = [1000]
        context_size = 2
        ratios = [2,3,4,5,6,7,8]
        c = 10
        hardness = {"g1": (0, c), "g2": (0, c)}
        delta = 0.1
        error_mean = 10
        error_std = 1
        for ratio in ratios:
            groups = {"g1": [i for i in range(0, ratio)], "g2": [i for i in range(ratio, arms)]}
            sensitive_group = {}
            for i in range(0,ratio):
                sensitive_group[i] = True
            for i in range(ratio,arms):
                sensitive_group[i] = False

            filename = "../experiments/%s/context_%s_ratio_%s_c_%s_delta_%s_error_%s_arms_%s_run_%s" % (args.exp, 
                        context_size, ratio, c, delta, error_mean, arms, args.run_name)
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))

            experiment = Experiment(arms,context_size,groups,
                        bandit_types=algorithms,deltas=[delta],
                        Ts=Ts,arm_type="uniform",
                        filename=filename,
                        cs=hardness,sensitive_group=sensitive_group,group_mean=error_mean, 
                        group_std=error_std)
            experiment.run_x_experiments(runs,seeds=seeds)
    if args.exp == "delta":
        arms = 10
        Ts = [1000]
        context_size = 2
        ratio = 5
        groups = {"g1": [i for i in range(0, ratio)], "g2": [i for i in range(ratio, arms)]}
        sensitive_group = {}
        for i in range(0,ratio):
            sensitive_group[i] = True
        for i in range(ratio,arms):
            sensitive_group[i] = False
        c = 10
        hardness = {"g1": (0, c), "g2": (0, c)}
        deltas = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
        error_mean = 10
        error_std = 1
        for delta in deltas:

            filename = "../experiments/%s/context_%s_ratio_%s_c_%s_delta_%s_error_%s_arms_%s_run_%s" % (args.exp, 
                        context_size, ratio, c, delta, error_mean, arms, args.run_name)
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))

            experiment = Experiment(arms,context_size,groups,
                        bandit_types=algorithms,deltas=[delta],
                        Ts=Ts,arm_type="uniform",
                        filename=filename,
                        cs=hardness,sensitive_group=sensitive_group,group_mean=error_mean, 
                        group_std=error_std)
            experiment.run_x_experiments(runs,seeds=seeds)
    if args.exp == "arms":
        num_arms = [i*10 for i in range(1,11)]
        Ts = [1000]
        context_size = 2
        c = 10
        hardness = {"g1": (0, c), "g2": (0, c)}
        delta = 0.1
        error_mean = 10
        error_std = 1
        for arms in num_arms:
            ratio = int(arms/2)
            groups = {"g1": [i for i in range(0, ratio)], "g2": [i for i in range(ratio, arms)]}
            sensitive_group = {}
            for i in range(0,ratio):
                sensitive_group[i] = True
            for i in range(ratio,arms):
                sensitive_group[i] = False

            filename = "../experiments/%s/context_%s_ratio_%s_c_%s_delta_%s_error_%s_arms_%s_run_%s" % (args.exp, 
                        context_size, ratio, c, delta, error_mean, arms, args.run_name)
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))

            experiment = Experiment(arms,context_size,groups,
                        bandit_types=algorithms,deltas=[delta],
                        Ts=Ts,arm_type="uniform",
                        filename=filename,
                        cs=hardness,sensitive_group=sensitive_group,group_mean=error_mean, 
                        group_std=error_std)
            experiment.run_x_experiments(runs,seeds=seeds)



