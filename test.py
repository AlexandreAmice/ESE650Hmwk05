import lake_env as lake_env
import numpy as np
from tester import Tester
from time import sleep
import gym
from copyTutorial import CopyPolicy
from pytorchModelTrainer import Policy
import torch
from torch.autograd import Variable


def select_action(state, policy):
    state = torch.from_numpy(state).type(torch.FloatTensor)
    state = policy(Variable(state))
    c = torch.distributions.Categorical(state)
    action = c.sample()
    log_prob = c.log_prob(action)

    if policy.policy_history.dim() != 0:
        policy.policy_history = torch.cat([policy.policy_history, log_prob])
    else:
        policy.policy_history = c.log_prob(action)
    return action

env = gym.make('CartPole-v1')
tester = Tester()

#model1 = torch.load('learnedPolicy')
model2 = torch.load('learnedPolicy2')
state = env.reset()
for i in range(1000):
    env.render()
    action = select_action(state, model2)
    state, reward, done, _ = env.step(action.data[0])
    if done:
        print(i)
        break

env.close()










