import numpy as np
from algorithms.TopIntervalContextualBandit import TopIntervalContextualBandit
from algorithms.IntervalChaining import IntervalChainingBandit
from algorithms.RandomBandit import RandomBandit
from algorithms.GroupFairParity import GroupFairParityBandit
from algorithms.GroupFairParityInterval import GroupFairParityIntervalBandit
from algorithms.GroupFairProportional import GroupFairProportionalBandit
from algorithms.GroupFairProportionalInterval import GroupFairProportionalIntervalBandit
from algorithms.GroupFairTopInterval import GroupFairTopIntervalBandit
from ContextualArm import GeneralContextualArm
from ErrorContextualArm import ErrorContextualArm
from RealContextualArm import RealDataContextualArm
from BanditDriver import BanditDriver
import collections
import pickle

ExperimentTuple = collections.namedtuple('ExperimentTuple', ['bandit','delta','T'])
Results = collections.namedtuple('Results', ['seed','experiments','rewards','opt_rewards','opt_real_rewards','pulled_arms','opt_arms','opt_real_arms','regret'])

class Experiment:

    def __init__(self,num_arms,context_size,groups,
        bandit_types=["TopInterval"],deltas=[0.5],
        Ts=[100],arm_type="guassian",betas=None,
        filename="../experiments/experiments.pkl",
        cs=None,sensitive_group=None,group_mean=None, 
        group_std=None, context_matrix=None, reward_matrix=None):
        self.num_arms = num_arms
        self.context_size = context_size
        self.groups = groups
        self.filename = filename
        self.bandit_types = bandit_types
        self.deltas = deltas
        self.Ts = Ts
        self.arm_type = arm_type
        self.betas = betas
        # cs is a dictionary from groups -> tuple (a, b), bounds of uniform distribution of beta for group
        self.cs = cs
        self.sensitive_group = sensitive_group
        self.arm_to_group = {}
        for group in self.groups:
            for arm in self.groups[group]:
                self.arm_to_group[arm] = group
        self.group_mean = group_mean
        self.group_std = group_std
        self.context_matrix = context_matrix
        self.reward_matrix = reward_matrix
        self.create_bandits()

    def create_arms(self):
        # if self.arm_type == "guassian":
        #     self.arms = [GeneralContextualArm(np.random.randn(self.context_size),self.context_size) for _ in range(self.num_arms)]
        if self.arm_type == "real":
            if self.context_matrix is None or self.reward_matrix is None:
                raise ValueError("real matrices need to be populated")
            self.arms = [None for _ in range(self.num_arms)]
            for group, idxs in self.groups.items():
                for idx in idxs:
                    self.arms[idx] = RealDataContextualArm(self.context_matrix[idx], self.reward_matrix[idx])
        elif self.arm_type == "uniform":
            if self.cs:
                self.arms = [None for _ in range(self.num_arms)]
                if self.group_mean is not None:
                    group_beta = np.random.rand(self.context_size) + self.group_mean

                for group, idxs in self.groups.items():
                    for idx in idxs:
                        # using (b-a)*Uniform(0,1) + a to get Uniform(a,b)
                        beta = (self.cs[group][1]-self.cs[group][0])*np.random.rand(self.context_size)+self.cs[group][0] + 30
                        if self.group_mean is None:
                            self.arms[idx] = GeneralContextualArm(beta,self.context_size)
                        else:
                            if self.sensitive_group[idx]:
                                self.arms[idx] = ErrorContextualArm(beta, self.context_size, group_beta)
                            else:
                                self.arms[idx] = GeneralContextualArm(beta,self.context_size)
            else:
                if group_mean is None:
                    self.arms = [GeneralContextualArm(np.random.rand(self.context_size),self.context_size) for _ in range(self.num_arms)]
                else:
                    raise ValueError("I am lazy - use the C's")
        elif self.arm_type == "specific":
            self.arms = [GeneralContextualArm(self.betas[i],self.context_size) for i in range(self.num_arms)]
        else:
            raise ValueError("Cannot initialize arms with type " + self.arm_type)

    def create_bandits(self):
        self.experiments = [ExperimentTuple(tp,delta,T) 
                                for tp in self.bandit_types 
                                for delta in self.deltas 
                                for T in self.Ts]
        self.bandits = []
        for experiment in self.experiments:
            if experiment.bandit == "TopInterval":
                self.bandits.append(TopIntervalContextualBandit(
                    self.num_arms, self.context_size, 
                    experiment.delta, experiment.T))
            elif experiment.bandit == "IntervalChaining":
                self.bandits.append(IntervalChainingBandit(
                    self.num_arms, self.context_size, 
                    experiment.delta, experiment.T))
            elif experiment.bandit == "Random":
                self.bandits.append(RandomBandit(
                    self.num_arms, self.context_size, 
                    experiment.delta, experiment.T))
            elif experiment.bandit == "GroupFairParity":
                self.bandits.append(GroupFairParityBandit(
                    self.num_arms, self.context_size, 
                    experiment.delta, experiment.T, self.groups))
            elif experiment.bandit == "GroupFairParityInterval":
                self.bandits.append(GroupFairParityIntervalBandit(
                    self.num_arms, self.context_size, 
                    experiment.delta, experiment.T, self.groups))
            elif experiment.bandit == "GroupFairProportional":
                self.bandits.append(GroupFairProportionalBandit(
                    self.num_arms, self.context_size, 
                    experiment.delta, experiment.T, self.groups))
            elif experiment.bandit == "GroupFairProportionalInterval":
                self.bandits.append(GroupFairProportionalIntervalBandit(
                    self.num_arms, self.context_size, 
                    experiment.delta, experiment.T, self.groups))
            elif experiment.bandit == "GroupFairTopInterval":
                assert(self.sensitive_group is not None)
                self.bandits.append(GroupFairTopIntervalBandit(self.num_arms, 
                    self.context_size, experiment.delta, experiment.T, 
                    self.groups, self.arm_to_group, self.sensitive_group))
            else:
                raise ValueError("No Bandit type of " + experiment.bandit)

    def run_experiment(self,seed=None):
        if seed is None:
            seed = np.random.random_integers(100000000)
        rewards = []
        opt_rewards = []
        opt_real_rewards = []
        pulled_arms = []
        opt_arms = []
        opt_real_arms = []
        regret = []
        for i in range(len(self.experiments)):
            experiment = self.experiments[i]
            bandit = self.bandits[i]
            print("RUNNING")
            print(experiment)
            print("\n")

            np.random.seed(seed)
            self.create_arms()
            driver = BanditDriver(self.arms,bandit)
            driver.complete_run()
            rewards.append(driver.get_received_rewards())
            opt_rewards.append(driver.get_optimal_rewards())
            pulled_arms.append(driver.get_pulled_arms())
            opt_arms.append(driver.get_optimal_arms())
            regret.append(driver.get_regret())
            opt_real_rewards.append(driver.get_optimal_real_rewards())
            opt_real_arms.append(driver.get_optimal_real_arms())

        return Results(seed,self.experiments,rewards,opt_rewards,opt_real_rewards,pulled_arms,opt_arms,opt_real_arms,regret)

    def run_x_experiments(self,x,save=True,seeds=None):
        self.experiment_results = []
        assert (seeds is None) or (len(seeds) == x)
        for i in range(x):
            print(i)
            if seeds is None:
                seed = None
            else:
                seed = seeds[i]
            self.create_bandits()
            result = self.run_experiment(seed)
            self.experiment_results.append(result)
            if save:
                with open(self.filename, 'wb') as f:
                    pickle.dump(self.__dict__,f)


# experiment = Experiment(3,2,{"Male": [0,1], "Female": [2]},["TopInterval", "IntervalChaining", "Random", "GroupFairParity", "GroupFairParityInterval", "GroupFairProportional", "GroupFairProportionalInterval"],[0.5,0.6])
# experiment.run_x_experiments(2)

