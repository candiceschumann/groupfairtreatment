class BanditDriver:

	def __init__(self,arms,bandit,T=1000):
		self.arms = arms,
		self.bandit = bandit
		self.T = T
