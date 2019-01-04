from algorithms.TopIntervalContextualBandit import TopIntervalContextualBandit
from ContextualArm import GeneralContextualArm

class BanditDriver:

	def __init__(self,arms,bandit):
		self.arms = arms
		self.bandit = bandit
		self.t = 0

	def run_for_t_times(self,t):
		if self.t + t > self.bandit.get_T():
			raise ValueError('Running for %d steps is more than the %d allowed') % (self.t + t, self.bandit.get_T())
		for i in range(t):
			self.t += 1
			# Get new context for this round from all of the arms.
			contexts = [self.arms[arm].get_new_context() for arm in range(len(self.arms))]
			# Pick an arm to pull and pull it
			arm = self.bandit.pick_arm(contexts, self.t)
			reward = self.arms[arm].pull_arm()
			# Update the bandit
			self.bandit.update_reward(arm, contexts[arm], reward)
			self.bandit.update_beta(arm)


	def complete_run(self):
		self.run_for_t_times(self.bandit.get_T() - self.t)
			


arms = [GeneralContextualArm([1,0,0,0],4), GeneralContextualArm([0,1,0,0],4), GeneralContextualArm([0,0,0.5,0.5],4)]
bandit = TopIntervalContextualBandit(len(arms), 4, 0.3, 100)
driver = BanditDriver(arms,bandit)
driver.complete_run()