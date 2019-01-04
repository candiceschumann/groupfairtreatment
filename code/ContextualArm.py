import numpy as np

''' The General Contextual Arm pulls context from a uniform distribution. '''
class GeneralContextualArm:

	'''Initialize the arm with a true beta and no pulls or context.'''
	def __init__(self,beta):
		self.beta = beta
		self.pulls = 0
		self.context = None

	'''Print out this arm in a more redable format'''
	def __str__(self):
		s = "Beta:\n"
		s += str(self.beta)
		s += "\nCurrent context:\n"
		s += str(self.context)
		s += "\nPulled %d times\n" % self.pulls
		return s

	'''Randomly pull a new context, save, and return'''
	def get_new_context(self):
		self.context = np.random.random(self.beta.shape[0])
		return self.context

	'''Return the most recent context'''
	def get_current_context(self):
		return self.context

	'''Pull the arm using the current context'''
	def pull_arm(self):
		if self.context is None:
			raise ValueError('trying to pull an arm with no context')
		self.pulls += 1
		return self.get_reward(self.context)

	'''Find the reward of a given context'''
	def get_reward(self,context):
		return np.dot(self.beta.T,context)[0]