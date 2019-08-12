from ContextualArm import GeneralContextualArm
import numpy as np

class RealDataContextualArm:

	'''Initialize the arm with a true reward and no pulls or context.'''
	def __init__(self, contexts, rewards):
		self.contexts = contexts
		self.rewards = rewards
		self.pulls = 0
		self.context_ind = None

	'''Print out this arm in a more readable format'''
	def __str__(self):
		s += "\nPulled %d times\n" % self.pulls
		return s

	'''Randomly pull a new context, save, and return'''
	def get_new_context(self):
		self.context_ind = np.random.uniform(0,len(self.rewards))
		return self.contexts[self.context_ind]

	'''Return the most recent context'''
	def get_current_context(self):
		return self.contexts[self.context_ind]

	'''Pull the arm using the current context'''
	def pull_arm(self):
		if self.context_ind is None:
			raise ValueError('trying to pull an arm with no context')
		self.pulls += 1
		return abs(self.rewards[self.context_ind] + np.random.randn())

	'''Find the reward of a given context'''
	def get_reward(self,context):
		return self.rewards[self.context_ind] # Kind of a hack but the current code only looks at current context

	def get_real_reward(self, context):
		return self.rewards[self.context_ind]
