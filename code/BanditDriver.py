import numpy as np
import matplotlib.pyplot as plt

class BanditDriver:

	def __init__(self,arms,bandit):
		self.arms = arms
		self.bandit = bandit
		self.t = 0
		self.rewards  = []
		self.opt_rewards = []
		self.which_arms = []
		self.opt_arms = []

	def run_for_t_times(self,t):
		if self.t + t > self.bandit.get_T():
			raise ValueError('Running for %d steps is more than the %d allowed') % (self.t + t, self.bandit.get_T())
		for i in range(t):
			self.t += 1
			# Get new context for this round from all of the arms.
			contexts = [self.arms[arm].get_new_context() for arm in range(len(self.arms))]
			true_rewards = [self.arms[arm].get_reward(contexts[arm]) for arm in range(len(self.arms))]
			self.opt_arms.append(np.argmax(true_rewards))
			self.opt_rewards.append(true_rewards[self.opt_arms[-1]])
			# Pick an arm to pull and pull it
			arm = self.bandit.pick_arm(contexts, self.t)
			self.which_arms.append(arm)
			reward = self.arms[arm].pull_arm()
			self.rewards.append(reward)
			# Update the bandit
			self.bandit.update_reward(arm, contexts[arm], reward)
			self.bandit.update_beta(arm)


	def complete_run(self):
		self.run_for_t_times(self.bandit.get_T() - self.t)

	def get_received_rewards(self):
		return self.rewards

	def get_optimal_rewards(self):
		return self.opt_rewards
	
	def get_pulled_arms(self):
		return self.which_arms

	def get_optimal_arms(self):
		return self.opt_arms

	def get_regret(self):
		regret = []
		for i in range(len(self.rewards)):
			regret.append(self.opt_rewards[i]-self.rewards[i])
		return regret
