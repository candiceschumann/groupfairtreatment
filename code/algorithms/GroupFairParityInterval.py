from GroupFairParity import *

class GroupFairParityIntervalBandit(GroupFairParityBandit):

	'''Return the set of arms that are chained to the given arm in the group'''
	def chained(self,arm,context,group):
		l_u = self.lower_uppers(context)
		chained = set([arm])
		curr_low = l_u[arm].lower
		curr_up = l_u[arm].upper
		for _ in range(2):
			for i in self.groups[group]:
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
		# Randomly pick a group
		group = self.groups_list[np.random.randint(len(self.groups))]
		print(group)
		if np.random.random() < self.eta(t):
			# Uniformly picking at random maintains individual fairness
			return np.random.choice(self.groups[group])
		else:
			l_u = self.lower_uppers(context)
			# In the group choose the arm with the 
			# highest upper confidence.
			arm = groups[group][0]
			for i in groups[group]:
				if l_u[i].upper > l_u[arm].upper:
					arm = i
			# Find the arms that are chained in that group
			arms = list(self.chained(arm,context,group))
			# Pick randomly from the chained arms
			return np.random.choice(arms)
