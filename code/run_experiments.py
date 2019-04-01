from Experiments import Experiment
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs Group Fair MAB experiments')
    parser.add_argument('exp', help="Experiment name")
    parser.add_argument('run_name', help="Run name (like number)")
    args = parser.parse_args()

# binary group different percentages
if args.exp == "GenderRatiosUniform":
	arms = 100
	runs = 1
	context = 2
	ratios = [1,5,10,25,50]
	deltas = [0.5,0.4,0.3,0.2,0.1]
	Ts = [10,20,30,40,50,60,70,80,90,100,250,500,750,1000,1500,2000,5000,10000]
	algorithms = ["TopInterval", "IntervalChaining", "Random", "GroupFairParity", "GroupFairParityInterval",
	              "GroupFairProportional", "GroupFairProportionalInterval"]
	for ratio in ratios:
		groups = {"g1": [i for i in range(0,ratio)], "g2": [i for i in range(ratio,arms)]}
		filename = "../experiments/%s/%s_%s" % (args.exp,ratio,args.run_name)
		if not os.path.exists(os.path.dirname(filename)):
			os.makedirs(os.path.dirname(filename))
		experiment = Experiment(arms, context, groups, algorithms, deltas, Ts, "uniform", filename=filename)
		experiment.run_x_experiments(runs)
elif args.exp == "GenderRatiosUniformSmaller":
	arms = 10
	runs = 20
	contexts = [2,3,4,5,6,7,8,9,10]
	ratios = [1,2,3,4,5]
	deltas = [0.5,0.4,0.3,0.2,0.1]
	Ts = [10,20,30,40,50,60,70,80,90,100,250,500,750,1000,1500,2000]
	algorithms = ["TopInterval", "IntervalChaining", "Random", "GroupFairParity", "GroupFairParityInterval",
	              "GroupFairProportional", "GroupFairProportionalInterval"]
	for context in contexts:
		for ratio in ratios:
			groups = {"g1": [i for i in range(0,ratio)], "g2": [i for i in range(ratio,arms)]}
			filename = "../experiments/%s/ratio_%s_context_%s_%s" % (args.exp,ratio,context,args.run_name)
			if not os.path.exists(os.path.dirname(filename)):
				os.makedirs(os.path.dirname(filename))
			experiment = Experiment(arms, context, groups, algorithms, deltas, Ts, "uniform", filename=filename)
			experiment.run_x_experiments(runs)
elif args.exp == "VaryContextSize":
	arms = 10
	deltas = [.1, .2, .3, .4, .5]
	cs = [1, 2, 5, 10]
	runs = 20
	ratios = list(range(2, 6))
	Ts = [2 * n * arms for n in range(1, 21)]
	algorithms = ["TopInterval", "IntervalChaining", "Random", "GroupFairParity", "GroupFairParityInterval",
	              "GroupFairProportional", "GroupFairProportionalInterval"]

	context_sizes = [2, 3, 4, 5]
	for context_size in context_sizes:
		for ratio in ratios:
			groups = {"g1": [i for i in range(0, ratio)], "g2": [i for i in range(ratio, arms)]}
			for c in cs:
				hardness = {"g1": (0, c), "g2": (0, c)}
				filename = "../experiments/%s/ratio_%s_context_%s_c_%s_%s" % (args.exp,ratio,context_size,c,args.run_name)
				if not os.path.exists(os.path.dirname(filename)):
					os.makedirs(os.path.dirname(filename))
				experiment = Experiment(arms, context_size, groups, algorithms, deltas, Ts, "uniform", filename=filename, cs=hardness)
				experiment.run_x_experiments(runs)
elif args.exp == 'VaryNumGroups':
	arms = 10
	deltas = [.1, .2, .3, .4, .5]
	cs = [1, 2, 5, 10]
	runs = 20
	Ts = [2 * n * arms for n in range(1, 21)]
	algorithms = ["TopInterval", "IntervalChaining", "Random", "GroupFairParity", "GroupFairParityInterval",
	              "GroupFairProportional", "GroupFairProportionalInterval"]
	context_size = 2

	def partition(lst, n):
		q, r = divmod(len(lst), n)
		indices = [q*i + min(i, r) for i in range(n+1)]
		return [lst[indices[i]:indices[i+1]] for i in range(n)]

	arm_indices = list(range(arms))
	for num_groups in range(1, 11):
		group_indices = partition(arm_indices, num_groups)
		groups = {"g{}".format(i+1): group_indices[i] for i in range(num_groups)}
		for c in cs:
			hardness = {"g{}".format(i+1): (0, c) for i in range(num_groups)}
			filename = "../experiments/%s/context_%s_numgrp_%s_c_%s_%s" % (args.exp,context_size,num_groups,c,args.run_name)
			if not os.path.exists(os.path.dirname(filename)):
				os.makedirs(os.path.dirname(filename))
			experiment = Experiment(arms, context_size, groups, algorithms, deltas, Ts, "uniform", filename=filename, cs=hardness)
			experiment.run_x_experiments(runs)
elif args.exp == 'VaryBounds':
	arms = 10
	deltas = [.1, .2, .3, .4, .5]
	cs = [1, 2, 5, 10]
	runs = 20
	ratios = list(range(2, 6))
	Ts = [2 * n * arms for n in range(1, 21)]
	algorithms = ["TopInterval", "IntervalChaining", "Random", "GroupFairParity", "GroupFairParityInterval",
	              "GroupFairProportional", "GroupFairProportionalInterval"]
	context_size = 2

	c_covered = [i/10 for i in range(4, 11)]
	for percentage in c_covered:
		for ratio in ratios:
			groups = {"g1": [i for i in range(0, ratio)], "g2": [i for i in range(ratio, arms)]}
			for c in cs:
				hardness = {"g1": (0, percentage * c), "g2": (c - percentage * c, c)}
				filename = "../experiments/%s/_context_%s_percentc_%s_c_%s_%s" % (args.exp, context_size, percentage, args.run_name)
				if not os.path.exists(os.path.dirname(filename)):
					os.makedirs(os.path.dirname(filename))
				experiment = Experiment(arms, context_size, groups, algorithms, deltas, Ts, "uniform",
				                        filename=filename, cs=hardness)
				experiment.run_x_experiments(runs)
