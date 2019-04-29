import lake_env as lake_env
import numpy as np
from tester import Tester
from time import sleep
import gym

env = gym.make('Stochastic-4x4-FrozenLake-v0')
tester = Tester()



value = tester.evaluate_policy(env, gamma=0.99, policy=np.ones(env.nS))
valueBestV, numIterV = tester.value_iteration(env, gamma =0.99)
print(value)
print(valueBestV)







