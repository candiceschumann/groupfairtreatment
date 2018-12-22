import numpy as np

class AbstractContextualBandit:

	def __init__(self, num_arms, delta, T, context_size):
		self.num_arms = num_arms
		self.context_size = context_size
		self.beta = None
		self.delta = delta
		self.T = T

	'''Probability of exploration at timestep t'''
	def eta(self, t):
		return 1.0 / (t ** (1.0/3))
	
