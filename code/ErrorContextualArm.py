from ContextualArm import GeneralContextualArm
import numpy as np


class ErrorContextualArm(GeneralContextualArm):

    '''Initialize the arm with a true beta and no pulls or context.'''
    def __init__(self, beta, context_size, group_beta):
        super().__init__(beta, context_size)
        self.group_beta = np.reshape(beta,(context_size,1))

    '''Find the reward of a given context'''
    def get_reward(self, context):
        return np.dot(self.beta.T,context)[0] + np.dot(self.group_beta.T,context)[0]

    def get_real_reward(self, context):
        return np.dot(self.beta.T, context)[0]