from algorithms.TopIntervalContextualBandit import *


class GroupFairTopIntervalBandit(TopIntervalContextualBandit):

	def __init__(self, num_arms, context_size, delta, T, groups, arm_to_group, sensitive_groups):
		super().__init__(num_arms, context_size, delta, T)
		self.groups = groups
		# arm_to_group is a dict from arm index -> group name
		self.arm_to_group = arm_to_group
		# sensitive_group is a dict from group name -> (bool, ind)
		# where bool is True if group is sensitive, ind is index of group
		self.sensitive_group = sensitive_groups
		self.group_beta = [None for _ in range(len(groups))]
		self.group_X = [None for _ in range(len(groups))]
		self.group_Y = [None for _ in range(len(groups))]

	def update_group_beta(self, arm):
		group_name = self.arm_to_group[arm]
		_, group = self.sensitive_group[group_name]
		if self.group_X[group] is not None:
			tmp1 = np.dot(self.group_X[group].T, self.group_X[group])
			tmp1 = np.linalg.pinv(tmp1)
			tmp2 = np.dot(self.group_X[group].T, self.group_Y[group])
			self.group_beta[group] = np.dot(tmp1, tmp2)

	def update_group_reward(self, arm, context, Y):
		group_name = self.arm_to_group[arm]
		_, group = self.sensitive_group[group_name]
		if self.group_X[group] is None or self.group_Y[group] is None:
			self.group_X[group] = np.array([context])
			self.group_Y[group] = np.array([[Y]])
		else:
			self.group_X[group] = np.append(self.group_X[group], [context], axis=0)
			self.group_Y[group] = np.append(self.group_Y[group], [[Y]], axis=0)

	def update(self, arm, context, Y):
		self.update_reward(arm, context, Y)
		self.update_beta(arm)
		self.update_group_reward(arm, context, Y)
		self.update_group_beta(arm)

	def pick_arm(self,context,t):
		if np.random.random() < self.eta(t):
			return np.random.randint(0, self.num_arms)
		else:
			

