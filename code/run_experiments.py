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
	algorithms = ["TopInterval", "IntervalChaining", "Random", "GroupFairParity", "GroupFairParityInterval", "GroupFairProportional", "GroupFairProportionalInterval"]
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
	algorithms = ["TopInterval", "IntervalChaining", "Random", "GroupFairParity", "GroupFairParityInterval", "GroupFairProportional", "GroupFairProportionalInterval"]
	for context in contexts:
		for ratio in ratios:
			groups = {"g1": [i for i in range(0,ratio)], "g2": [i for i in range(ratio,arms)]}
			filename = "../experiments/%s/ratio_%s_context_%s_%s" % (args.exp,ratio,context,args.run_name)
			if not os.path.exists(os.path.dirname(filename)):
				os.makedirs(os.path.dirname(filename))
			experiment = Experiment(arms, context, groups, algorithms, deltas, Ts, "uniform", filename=filename)
			experiment.run_x_experiments(runs)