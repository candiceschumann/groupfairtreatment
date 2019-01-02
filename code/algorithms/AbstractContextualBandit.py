import numpy as np
from scipy.stats import norm
import math

class AbstractContextualBandit:

	def __init__(self, num_arms, delta, T, context_size):
		self.num_arms = num_arms
		self.delta = delta
		self.T = T
		self.context_size = context_size
		self.beta = [None for _ in range(num_arms)]
		self.X = [None for _ in range(num_arms)]
		self.Y = [None for _ in range(num_arms)]

	def __str__(self):
		s = '\nBandit object\n'
		s += 'Num arms: %d\n' % self.num_arms
		s += 'Context size: %d\n' % self.context_size
		s += 'Context:\n'
		s += str(self.X)
		s += '\nRewards:\n'
		s += str(self.Y)
		s += '\nContext weights:\n'
		s += str(self.beta)
		s += '\n'
		return s

	'''Probability of exploration at timestep t'''
	def eta(self, t):
		return 1.0 / (t ** (1.0/3))

	'''Update the context and received reward for the given arm'''
	def update_reward(self, arm, context, Y):
		if self.X[arm] is None or self.Y[arm] is None:
			self.X[arm] = np.array([context])
			self.Y[arm] = np.array([[Y]])
		else:
			self.X[arm] = np.append(self.X[arm], [context], axis=0)
			self.Y[arm] = np.append(self.Y[arm], [[Y]], axis=0)

	'''Update the weights on each context for a given arm'''
	def update_beta(self, arm):
		if self.X[arm] is not None:
			# (X'X)^-1
			tmp1 = np.dot(self.X[arm].T,self.X[arm])
			tmp1 = np.linalg.inv(tmp1)
			# X'Y
			tmp2 = np.dot(self.X[arm].T, self.Y[arm])
			#(X'X)^-1X'Y
			self.beta[arm] = np.dot(tmp1,tmp2)

	'''Find the score for each arm given the empirical beta's calculated 
	and the given context (x) for each arm.'''
	def estimate_rewards(self, X):
		Y = [None for _ in range(self.num_arms)]
		for arm in range(self.num_arms):
			Y[arm] = self.estimate_reward(arm,X[arm])
		return Y

	'''Estimate a single arm'''
	def estimate_reward(self,arm, X):
		if self.beta[arm] is not None:
			return np.dot(self.beta[arm].T,X)[0]
		else:
			return None

	'''Find the standard deviation for the margin'''
	def arm_sigma(self, arm, context):
		if self.X[arm] is None:
			return None
		else:
			tmp = np.dot(self.X[arm].T, self.X[arm])
			tmp = np.linalg.inv(tmp)
			tmp = np.dot(context, tmp)
			return np.dot(tmp, context.T)

	def arm_confidence(self, arm, context):
		if self.X[arm] is None:
			return float('inf')
		else:
			sigma = abs(self.arm_sigma(arm, context))
			intv = self.delta / (2 * self.num_arms * self.T)
			return abs(norm.ppf(intv,0,sigma))

	def lower_upper(self, arm, context):
		Y = self.estimate_reward(arm, context)
		w = self.arm_confidence(arm, context)
		if math.isinf(w):
			return [float('-inf'), float('inf')]
		else:
			return [Y-w,Y+w]


num_arms = 3
delta = 0.5
T = 10
context_size = 4

betas = [np.array([[1,0,0,0]]).T, np.array([[0,1,0,0]]).T, np.array([[0,0,0.5,0.5]]).T]

bandit = AbstractContextualBandit(num_arms, delta, T, context_size)
print(bandit)
for t in range(20):
	context = [np.random.rand(context_size) for i in range(num_arms)]
	print(bandit.estimate_rewards(context))
	print(bandit.arm_sigma(0,context[0]))
	print(bandit.arm_confidence(0,context[0]))
	print(bandit.lower_upper(0,context[0]))
	bandit.update_reward(0, context[0], np.dot(betas[0].T, context[0])[0])
	bandit.update_beta(0)
	print(bandit)



