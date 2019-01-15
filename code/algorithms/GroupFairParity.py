from GroupFairProportional import *

class GroupFairParityBandit(GroupFairProportionalBandit):

	def __init__(self, num_arms, context_size, delta, T, groups):
		super().__init__(num_arms, context_size, delta, T,groups)
		self.groups_list = list(groups.keys())

	'''Return an arm given the context and timestep.'''
	def pick_arm(self,context,t):
		# Randomly pick a group
		group = self.groups_list[np.random.randint(len(self.groups))]
		if np.random.random() < self.eta(t):
			return np.random.choice(self.groups[group])
		else:
			l_u = self.lower_uppers(context)
			# In the group choose the arm with the 
			# highest upper confidence.
			arm = groups[group][0]
			for i in groups[group]:
				if l_u[i].upper > l_u[arm].upper:
					arm = i
			return arm

groups = {'Female': [1], 'Male': [0,2]}
bandit = GroupFairParityBandit(3,4,0.5,10,groups)
context = [np.random.random(4) for _ in range(3)]
print(bandit.pick_arm(context,100))
