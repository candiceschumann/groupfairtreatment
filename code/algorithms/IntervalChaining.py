from TopIntervalContextualBandit import *

class IntervalChainingBandit(TopIntervalContextualBandit):

	'''Return the set of arms that are chained to the given arm'''
	def chained(self,arm,context):
		l_u = self.lower_uppers(context)
		chained = set([arm])
		curr_low = l_u[arm].lower
		curr_up = l_u[arm].upper
		for _ in range(2):
			for i in range(self.num_arms):
				# Is the lower bound in the current bound?
				if (l_u[i].lower >= curr_low) and (l_u[i].lower <= curr_up):
					if(l_u[i].upper > curr_up):
						curr_up = l_u[i].upper
					chained.add(i)
				# Is the upper bound in the current bound?
				elif (l_u[i].upper >= curr_low) and (l_u[i].upper <= curr_up):
					if l_u[i].lower < curr_low:
						curr_low = l_u[i].upper
					chained.add(i)
				# Does the new bound surround the current bound?
				elif (l_u[i].upper >= curr_up) and (l_u[i].lower <= curr_low):
					curr_low = l_u[i].lower
					curr_up = l_u[i].upper
					chained.add(i)
		return chained
	
	'''Return an arm given the context and timestep.'''
	def pick_arm(self,context,t):
		if np.random.random() < self.eta(t):
			return np.random.randint(0,self.num_arms)
		else:
			l_u = self.lower_uppers(context)
			arm = 0
			for i in range(1,self.num_arms):
				if l_u[i].upper > l_u[arm].upper:
					arm = i
			arms = list(self.chained(arm,context))
			return np.random.choice(arms)


