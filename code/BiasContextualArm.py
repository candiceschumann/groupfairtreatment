from ErrorContextualArm import ErrorContextualArm
import numpy as np


class BiasContextualArm(ErrorContextualArm):

	'''Initialize the arm with a true beta and no pulls or context.'''
	def __init__(self, beta, context_size, group_bias, error_mean=0, error_std=1):
		super().__init__(beta, context_size, error_mean, error_std)
		self.group_bias = group_bias

	'''Find the reward of a given context'''
	def get_reward(self, context):
		return np.dot(self.beta.T,context)[0] + np.random.normal(self.error_mean, self.error_std) \
		       + np.dot(self.group_bias.T, context)[0]