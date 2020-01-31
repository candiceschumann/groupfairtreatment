import numpy as np
# import matplotlib.pyplot as plt

class BanditDriver:

	def __init__(self,arms,bandit):
		self.arms = arms
		self.bandit = bandit
		self.t = 0
		self.rewards  = []
		self.real_rewards = []
		self.opt_rewards = []
		self.opt_real_rewards = []
		self.which_arms = []
		self.opt_arms = []
		self.opt_real_arms = []

	def run_for_t_times(self,t):
		if self.t + t > self.bandit.get_T():
			raise ValueError('Running for %d steps is more than the %d allowed') % (self.t + t, self.bandit.get_T())
		reward_sum = 0
		real_reward_sum = 0
		opt_reward_sum = 0
		opt_real_reward_sum = 0
		for i in range(t):

			self.t += 1
			# Get new context for this round from all of the arms.
			contexts = [self.arms[arm].get_new_context() for arm in range(len(self.arms))]
			true_rewards = [self.arms[arm].get_reward(contexts[arm]) for arm in range(len(self.arms))]
			# print (true_rewards)
			unbiased_rewards = [self.arms[arm].get_real_reward(contexts[arm]) for arm in range(len(self.arms))]
			# print (unbiased_rewards)
			self.opt_arms.append(np.argmax(true_rewards))
			self.opt_real_arms.append(np.argmax(unbiased_rewards))
			self.opt_rewards.append(true_rewards[self.opt_arms[-1]])
			opt_reward_sum += self.opt_rewards[-1]
			self.opt_real_rewards.append(unbiased_rewards[self.opt_real_arms[-1]])
			opt_real_reward_sum += self.opt_real_rewards[-1]
			print("round " + str(i))
			print("opt_real_rewards " + str(self.opt_real_rewards[-1]))
			print("opt reward " + str(self.opt_rewards[-1]))
			# Pick an arm to pull and pull it
			arm = self.bandit.pick_arm(contexts, self.t)
			try:
				expected = self.bandit.upper(arm, contexts[arm])
				# print(self.bandit.uppers(contexts))
			except:
				expected = self.bandit.lower_upper(arm, contexts[arm]).upper
			print('expected ' + str(expected))

			self.which_arms.append(arm)
			reward = self.arms[arm].pull_arm()
			print ("reward " + str(reward))
			reward_sum += reward
			self.rewards.append(reward)
			real_reward = unbiased_rewards[arm]
			print ("real_reward" + str(unbiased_rewards[arm]))
			real_reward_sum += real_reward
			self.real_rewards.append(real_reward)
			# Update the bandit
			self.bandit.update(arm, contexts[arm], reward)
		print("reward sum " + str(reward_sum/500))
		print("real reward sum " + str(real_reward_sum/500))
		print(" opt reward_sum " + str(opt_reward_sum/500))
		print(" opt real reward sum " + str(opt_real_reward_sum/500))
		print(" average real regret " + str(np.mean(np.array(self.opt_real_rewards) - np.array(self.real_rewards))))
		print( "average regret " + str(np.mean(np.array(self.opt_rewards) - np.array(self.rewards))))
		plt.plot(np.array(self.opt_real_rewards) - np.array(self.real_rewards))
		plt.plot(np.array(self.opt_rewards) - np.array(self.rewards))
		# plt.show()


	def complete_run(self):
		self.run_for_t_times(self.bandit.get_T() - self.t)

	def get_received_rewards(self):
		return self.rewards

	def get_real_rewards(self):
		return self.real_rewards

	def get_optimal_rewards(self):
		return self.opt_rewards

	def get_optimal_real_rewards(self):
		return self.opt_real_rewards
	
	def get_pulled_arms(self):
		return self.which_arms

	def get_optimal_arms(self):
		return self.opt_arms

	def get_optimal_real_arms(self):
		return self.opt_real_arms

	def get_regret(self):
		regret = []
		for i in range(len(self.rewards)):
			regret.append(self.opt_real_rewards[i]-self.rewards[i])
		return regret
