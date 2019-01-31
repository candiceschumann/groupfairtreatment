import numpy as np
from algorithms.TopIntervalContextualBandit import TopIntervalContextualBandit
from algorithms.IntervalChaining import IntervalChainingBandit
from algorithms.RandomBandit import RandomBandit
from algorithms.GroupFairParity import GroupFairParityBandit
from algorithms.GroupFairParityInterval import GroupFairParityIntervalBandit
from algorithms.GroupFairProportional import GroupFairProportionalBandit
from algorithms.GroupFairProportionalInterval import GroupFairProportionalIntervalBandit
from ContextualArm import GeneralContextualArm
from BanditDriver import BanditDriver
import collections
import pickle

ExperimentTuple = collections.namedtuple('ExperimentTuple', ['bandit','delta','T'])
Results = collections.namedtuple('Results', ['seed','rewards','opt_rewards','pulled_arms','opt_arms','regret'])

class Experiment:

	def __init__(self,num_arms,context_size,groups,bandit_types=["TopInterval"],deltas=[0.5],Ts=[100],arm_type="guassian",betas=None,filename="../experiments/experiments.pkl"):
		self.num_arms = num_arms
		self.context_size = context_size
		self.groups = groups
		self.filename = filename
		self.bandit_types = bandit_types
		self.deltas = deltas
		self.Ts = Ts
		self.arm_type = arm_type
		self.betas = betas
		self.create_bandits()

	def create_arms(self):
		if self.arm_type == "guassian":
			self.arms = [GeneralContextualArm(np.random.randn(self.context_size),self.context_size) for _ in range(self.num_arms)]
		elif self.arm_type == "uniform":
			self.arms = [GeneralContextualArm(np.random.rand(self.context_size),self.context_size) for _ in range(self.num_arms)]
		elif self.arm_type == "specific":
			self.arms = [GeneralContextualArm(self.betas[i],self.context_size) for i in range(self.num_arms)]
		else:
			raise ValueError("Cannot innitialize arms with type " + self.arm_type)

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
			else:
				raise ValueError("No Bandit type of " + experiment.bandit)

	def run_experiment(self,seed=None):
		if seed is None:
			seed = np.random.random_integers(100000000)
		rewards = []
		opt_rewards = []
		pulled_arms = []
		opt_arms = []
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

		return Results(seed,rewards,opt_rewards,pulled_arms,opt_arms,regret)

	def run_x_experiments(self,x,save=True,seeds=None):
		self.experiment_results = []
		for i in range(x):
			if seeds is None:
				seed = None
			else:
				seed = seeds[i]
			self.experiment_results.append(self.run_experiment(seed))
			with open(self.filename, 'wb') as f:
				pickle.dump(self.__dict__,f)


# experiment = Experiment(3,2,{"Male": [0,1], "Female": [2]},["TopInterval", "IntervalChaining", "Random", "GroupFairParity", "GroupFairParityInterval", "GroupFairProportional", "GroupFairProportionalInterval"],[0.5,0.6])
# experiment.run_x_experiments(2)

