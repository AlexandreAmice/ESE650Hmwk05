import torch

from torch.autograd import Variable
from torch import optim
import torch.distributions
import gym
import numpy as np
from torch import nn


env = gym.make('CartPole-v1')
gamma = 0.8
learning_rate = 0.01


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.fc1 = nn.Linear(self.state_space, 128, bias=False)
        self.fc2 = nn.Linear(128, 128, bias=False)
        self.fc3 = nn.Linear(128, self.action_space, bias=False)
        self.gamma = gamma

        self.policy_history = Variable(torch.Tensor([]))

        self.reward_episode = []
        self.reward_history = []
        self.loss_history = []

    def forward(self, x):
        model = nn.Sequential(
            self.fc1,
            nn.Dropout(p=0.6),
            nn.ReLU(),

            self.fc2,
            nn.Dropout(p=0.4),
            nn.ReLU(),

            self.fc3,
            nn.Softmax(dim=-1)
        )
        return model(x)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

def select_action(state):
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

def update_policy():
    R = 0
    rewards = []

    for r in policy.reward_episode[::-1]:
        R = r + policy.gamma*R
        rewards.insert(0,R)

    #Scale rewards
    rewards = torch.FloatTensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

    #Calculate loss
    loss = (torch.sum(torch.mul(policy.policy_history, Variable(rewards)).mul(-1), -1))

    #Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #update policy
    policy.loss_history.append(loss.data[0])
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.policy_history = Variable(torch.Tensor())
    policy.reward_episode = []



if __name__ == '__main__':
    running_reward = 10
    numEpisodes = 10000
    for episode in range(numEpisodes):
        state = env.reset()
        done = False

        for time in range(1000):
            action = select_action(state)
            state, reward, done, _ = env.step(action.data[0])

            policy.reward_episode.append(reward)
            if done:
                break

        running_reward = (running_reward*0.99) + (time * 0.01)

        update_policy()

        if episode % 50 == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(episode, time, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(running_reward,
                                                                                                        time))
            break

    torch.save(policy, "learnedPolicy")











