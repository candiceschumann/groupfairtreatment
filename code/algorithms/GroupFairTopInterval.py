from algorithms.TopIntervalContextualBandit import *


class GroupFairTopIntervalBandit(TopIntervalContextualBandit):

	def __init__(self, num_arms, context_size, delta, T, groups, arm_to_group, sensitive_group):
		super().__init__(num_arms, context_size, delta, T)
		self.groups = groups
		# arm_to_group is a dict from arm index -> group name
		self.arm_to_group = arm_to_group
		# sensitive_group is a dict from group name -> bool
		# where bool is True if group is sensitive. ind 0 is always not sensitive group, ind 1 is always sensitive
		self.sensitive_group = sensitive_group
		self.group_beta = [None for _ in range(2)]
		self.group_X = [None for _ in range(2)]
		self.group_Y = [None for _ in range(2)]
		self.sens_group_size = 0
		self.nonsens_group_size = 0
		# Count number of arms for non sensitive group and sensitive group.
		for i in range(num_arms):
			if self.sensitive_group[i]:
				self.sens_group_size += 1
			else:
				self.nonsens_group_size += 1
		assert (self.sens_group_size > 0) and (self.nonsens_group_size > 0)

	'''takes an arm index and returns the bool which indicates whether the arm's group is sensitive'''
	def is_sensitive_group(self, arm):
		return self.sensitive_group[arm]

	'''update beta for a given arm's group'''
	def update_group_beta(self, arm):
		group = self.is_sensitive_group(arm)
		if group:
			group = 1
		else:
			group = 0
		if self.group_X[group] is not None:
			tmp1 = np.dot(self.group_X[group].T, self.group_X[group])
			tmp1 = np.linalg.pinv(tmp1)
			tmp2 = np.dot(self.group_X[group].T, self.group_Y[group])
			self.group_beta[group] = np.dot(tmp1, tmp2)

	'''update context & reward for a given arm's group, given context and reward observed'''
	def update_group_reward(self, arm, context, Y):
		group = self.is_sensitive_group(arm)
		if group:
			group = 1
		else:
			group = 0
		if self.group_X[group] is None or self.group_Y[group] is None:
			self.group_X[group] = np.array([context])
			self.group_Y[group] = np.array([[Y]])
		else:
			self.group_X[group] = np.append(self.group_X[group], [context], axis=0)
			self.group_Y[group] = np.append(self.group_Y[group], [[Y]], axis=0)

	'''update the arm and group rewards and betas'''
	def update(self, arm, context, Y):
		self.update_reward(arm, context, Y)
		self.update_beta(arm)
		self.update_group_reward(arm, context, Y)
		self.update_group_beta(arm)

	'''find the group standard deviation for the margin'''
	def group_sigma(self, group, context):
		if self.group_X[group] is None:
			return None
		else:
			tmp = np.dot(self.group_X[group].T, self.group_X[group])
			tmp = np.linalg.pinv(tmp)
			tmp = np.dot(context, tmp)
			return np.dot(tmp, context.T)

	'''using given delta and a qualtile function distribution fin the confidence interval of a given group'''
	def group_confidence(self, group, context):
		if self.group_X[group] is None:
			return float('inf')
		else:
			group_size = self.sens_group_size if group else self.nonsens_group_size
			sigma = abs(self.group_sigma(group, context))
			intv = self.delta / (2 * (self.num_arms/group_size) * self.T)
			return abs(norm.ppf(intv, 0, sigma))

	'''return the upper confidence bound of a single arm, given the context'''
	def upper(self, arm, context):
		Y = self.estimate_reward(arm, context)
		w = self.arm_confidence(arm, context)
		group = self.is_sensitive_group(arm)
		if group:
			group = 1
		else:
			group = 0
		sens_confidence = self.group_confidence(0, context)
		nonsens_confidence = self.group_confidence(1, context)
		# if group:
		if math.isinf(w) or Y is None:
			return float('inf')
		if group == 0:
			return Y+w
		else:
			if self.group_beta[0] is None or self.group_beta[1] is None:
				return float('inf')
			Y_sens = np.dot(self.group_beta[0].T, context)
			Y_nonsens = np.dot(self.group_beta[1].T, context)
			return (Y+w - Y_sens + sens_confidence + Y_nonsens + nonsens_confidence)[0]
			# return (Y + w - np.dot(self.group_beta[group].T, context) + sens_confidence + \
			#        np.dot(np.ones_like(self.group_beta[group]).T, context)*20)[0]
		# else:
		# 	if math.isinf(w) or Y is None:
		# 		return float('inf')
		# 	else:
		# 		return Y + w + sens_confidence + nonsens_confidence

	def uppers(self, X):
		u = [None for _ in range(self.num_arms)]
		for arm in range(self.num_arms):
			u[arm] = self.upper(arm, X[arm])
		return u

	def pick_arm(self,context,t):
		if np.random.random() < self.eta(t):
			return np.random.randint(0, self.num_arms)
		else:
			u = self.uppers(context)
			return np.argmax(u)



