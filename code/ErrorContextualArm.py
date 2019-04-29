from ContextualArm import GeneralContextualArm
import numpy as np


class ErrorContextualArm(GeneralContextualArm):

	'''Initialize the arm with a true beta and no pulls or context.'''
	def __init__(self, beta, context_size, error_mean=0, error_std=1):
		super().__init__(beta, context_size)
		self.error_mean = error_mean
		self.error_std = error_std

	'''Find the reward of a given context'''
	def get_reward(self, context):
		return np.dot(self.beta.T,context)[0] + np.random.normal(self.error_mean, self.error_std)