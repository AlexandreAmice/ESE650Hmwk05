import numpy as np
import torch
import gym
from pytorchModelTrainer import Policy
from torch.autograd import Variable


class Tester(object):

    def __init__(self):
        """
        Initialize the Tester object by loading your model.
        """
        self.policy = torch.load('learnedPolicy')


    def evaluate_policy(self, env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
        """Evaluate the value of a policy.

        See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
        book.

        http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

        Parameters
        ----------
        env: gym.core.Environment
          The environment to compute value iteration for. Must have nS,
          nA, and P as attributes.
        gamma: float
          Discount factor, must be in range [0, 1)
        policy: np.array
          The policy to evaluate. Maps states to actions.
        max_iterations: int
          The maximum number of iterations to run before stopping.
        tol: float
          Determines when value function has converged.

        Returns
        -------
        np.ndarray
          The value for the given policy

        number of iterations
        """
        Ppi = np.zeros((env.nS, env.nS))
        Rpi = np.zeros(env.nS)
        policy = np.reshape(policy, (16,))

        #Set up matrices
        for s in range(0, env.nS):
            a = policy[s]
            for item in env.P[s][a]:
                # item 0 = p(s'|s,a); #item 1 = s'; item 2 = reward; item 4 = isterminal
                Ppi[s,item[1]] += item[0]
                Rpi[s] += item[0] * item[2]

        value = np.zeros(env.nS)
        valueOld = -float('Inf')*np.ones((env.nS, 1))
        numIter = 0
        while np.any(np.abs(value-valueOld)) > tol:
            valueOld = value
            value = Rpi + gamma*np.matmul(Ppi, valueOld)
            numIter += 1


        return value, numIter

    def policy_iteration(self, env, gamma, max_iterations=int(1e3), tol=1e-3):
        """Runs policy iteration.

        See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
        book.

        http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

        You should use the improve_policy and evaluate_policy methods to
        implement this method.

        Parameters
        ----------
        env: gym.core.Environment
          The environment to compute value iteration for. Must have nS,
          nA, and P as attributes.
        gamma: float
          Discount factor, must be in range [0, 1)
        max_iterations: int
          The maximum number of iterations to run before stopping.
        tol: float
          Determines when value function has converged.

        Returns
        -------
        (np.ndarray, np.ndarray, int, int)
           Returns optimal policy, value function, number of policy
           improvement iterations, and number of value iterations.
        """
        #Setup matrices
        Pssa = np.zeros((env.nS, env.nS, env.nA))
        Rsa = np.zeros((env.nS, env.nA))
        for a in range(0, env.nA):
            for s in range(0, env.nS):
                for item in env.P[s][a]:
                    # item 0 = p(s'|s,a); #item 1 = s'; item 2 = reward; item 4 = isterminal
                    Pssa[s,item[1],a] += item[0]
                    Rsa[s,a] += item[0] * item[2]

        numPolicyImprov = 0
        valueOld = -float('Inf') * np.ones((env.nS, 1))
        policy = np.zeros(env.nS)
        (valueNew, numValIter) = self.evaluate_policy(env, gamma, policy)
        while np.any(abs(valueNew-valueOld)) > tol:
            valueOld = valueNew
            qpi = np.zeros((env.nS, env.nA))
            for a in range(0,env.nA):
                qpi[:,a] = Rsa[:,a] + gamma * np.matmul(Pssa[:,:,a], valueOld)
            policy = np.argmax(qpi, 1)
            (valueNew, valIter) = self.evaluate_policy(env, gamma, policy)
            numValIter += valIter
            numPolicyImprov += 1




        return policy, valueNew, numPolicyImprov, numValIter

    def value_iteration(self, env, gamma, max_iterations=int(1e3), tol=1e-3):
        """Runs value iteration for a given gamma and environment.

        See page 90 (pg 108 pdf) of the Sutton and Barto Second Edition
        book.

        http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

        Parameters
        ----------
        env: gym.core.Environment
          The environment to compute value iteration for. Must have nS,
          nA, and P as attributes.
        gamma: float
          Discount factor, must be in range [0, 1)
        max_iterations: int
          The maximum number of iterations to run before stopping.
        tol: float
          Determines when value function has converged.

        Returns
        -------
        np.ndarray, iteration
          The value function and the number of iterations it took to converge.
        """
        valueNew = np.zeros((env.nS))
        valueOld = -float('Inf')*np.ones((env.nS, 1))
        numIter = 0
        while np.any(abs(valueNew-valueOld) > tol) and numIter < max_iterations:
            numIter += 1
            options = np.zeros((env.nS, env.nA))
            valueOld = valueNew
            for a in range(0,env.nA):
                Pssa = np.zeros((env.nS, env.nS))
                Rsa = np.zeros(env.nS)
                for s in range(0,env.nS):
                    for item in env.P[s][a]:
                        #item 0 = p(s'|s,a); #item 1 = s'; item 2 = reward; item 4 = isterminal
                        Pssa[s][item[1]] += item[0]
                        Rsa[s] += item[0]*item[2]
                options[:,a] = Rsa + gamma*np.matmul(Pssa,valueOld)
            valueNew = np.amax(options, 1)
        return valueNew, numIter

    def policy_gradient_test(self, state):
        """
        Parameters
        ----------
        state: np.ndarray
            The state from the CartPole gym environment.
        Returns
        ------
        np.ndarray
            The action in this state according to the trained policy.
        """
        # TODO. Your Code goes here.
        return 0

    def select_action(self, state):
        state = torch.from_numpy(state).type(torch.FloatTensor)
        state = self.policy(Variable(state))
        c = torch.distributions.Categorical(state)
        action = c.sample()
        log_prob = c.log_prob(action)

        if self.policy.policy_history.dim() != 0:
            self.policy.policy_history = torch.cat([self.policy.policy_history, log_prob])
        else:
            self.policy.policy_history = c.log_prob(action)
        return action