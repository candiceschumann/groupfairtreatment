from Experiments import *
from BanditDriver import BanditDriver
import pickle
import argparse
import collections

SingleResult = collections.namedtuple('SingleResult', 
    ['name', 'experiment', 'rewards', 'opt_real_rewards', 
    'pulled_arms', 'opt_real_arms', 'regret'])

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
