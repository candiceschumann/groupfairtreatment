from algorithms.TopIntervalContextualBandit import *

class RandomBandit(TopIntervalContextualBandit):

	def eta(self,t):
		return 1

	