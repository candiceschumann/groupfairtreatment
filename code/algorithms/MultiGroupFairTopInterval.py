from algorithms.GroupFairTopInterval import *


class MultiGroupFairTopIntervalBandit(GroupFairTopIntervalBandit):

    def __init__(self, num_arms, context_size, delta, T, groups, arm_to_group, sensitive_group, mu):
        super().__init__(num_arms, context_size, delta, T, groups, arm_to_group, sensitive_group)
        self.mu = mu
        self.group_beta = {}
        self.group_X = {}
        self.group_Y = {}

    '''update beta for a given arm's group'''
    def update_group_beta(self, arm):
        group = self.arm_to_group[arm]
        if group in self.group_X:
            tmp1 = np.dot(self.group_X[group].T, self.group_X[group])
            tmp1 = np.linalg.pinv(tmp1)
            tmp2 = np.dot(self.group_X[group].T, self.group_Y[group])
            self.group_beta[group] = np.dot(tmp1, tmp2)

    '''update context & reward for a given arm's group, given context and reward observed'''
    def update_group_reward(self, arm, context, Y):
        group = self.arm_to_group[arm]
        if  group not in self.group_X or group not in self.group_Y:
            self.group_X[group] = np.array([context])
            self.group_Y[group] = np.array([[Y]])
        else:
            self.group_X[group] = np.append(self.group_X[group], [context], axis=0)
            self.group_Y[group] = np.append(self.group_Y[group], [[Y]], axis=0)

    '''using given delta and a qualtile function distribution fin the confidence interval of a given group'''
    def group_confidence(self, group, context):
        if group not in self.group_X:
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
        group = self.arm_to_group[arm]
        group_confidence = self.group_confidence(group, context)
        if math.isinf(w) or Y is None or math.isinf(group_confidence) or self.group_beta[group] is None:
            return float('inf')
        else:
            Y_group = np.dot(self.group_beta[group].T, context)
            return (Y+w + self.mu - Y_group + group_confidence)[0]



