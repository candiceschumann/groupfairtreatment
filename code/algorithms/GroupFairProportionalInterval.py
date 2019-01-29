from algorithms.GroupFairParityInterval import *

class GroupFairProportionalIntervalBandit(GroupFairParityIntervalBandit):

	'''Return an arm given the context and timestep.'''
	def pick_arm(self,context,t):
		# Randomly picking an arm maintains proportions and individual fairness
		if np.random.random() < self.eta(t):
			return np.random.randint(0,self.num_arms)
		else:
			l_u = self.lower_uppers(context)
			# Pick a group according to proportions
			rnd = np.random.random()
			tot = 0
			for gp in self.groups:
				tot += len(self.groups[gp])
				if rnd < (tot * 1.0) / self.num_arms:
					group = gp
					break
			# In that group choose the arm with the 
			# highest upper confidence.
			arm = self.groups[group][0]
			for i in self.groups[group]:
				if l_u[i].upper > l_u[arm].upper:
					arm = i
			# Find the arms that are chained in that group
			arms = list(self.chained(arm,context,group))
			# Pick randomly from the chained arms
			return np.random.choice(arms)
