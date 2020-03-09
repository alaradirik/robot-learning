import math

import gym
import numpy as np
import matplotlib.pyplot as plt


""" 
CartPole-v1 is considered "solved" when the agent obtains a reward of at
least 400 over 50 consecutive episodes.
"""

MAX_STEPS = 500

class QLearningAgent():
    def __init__(self, episodes=1000, lr=1, epsilon=1, gamma=1.0, lr_decay=0.99, epsilon_decay=0.99):
        """
        Construct a new Q-learning agent.

        :param buckets: Discretization buckets for position, velocity, angle, angular velocity
        :param episodes: Number of training episodes 
        :param lr: Learning rate
        :param epsilon: Exploration rate
        :param gamma: Reward discount rate
        :param lr_decay: Learning rate decay factor
        :param epsilon_decay: Epsilon decay factor

        :return: returns nothing
        """

        self.buckets = (1, 1, 6, 12)
        self.reward_target = 400
        self.episodes = episodes         
        self.lr = lr
        self.gamma = gamma 
        self.epsilon = epsilon
        self.lr_decay = lr_decay
        self.epsilon_decay = epsilon_decay
        
        # Load CartPole V1 environment
        self.env = gym.make('CartPole-v1')
        self.env._max_episode_steps = MAX_STEPS
        
        # Initialize Q-table with zeros
        # Rows corresponds to states, columns correspond to actions
        self.q_table = np.zeros(self.buckets + (self.env.action_space.n,))

    def discretize_observation(self, obs):
        # CartPole environment's low and high values for velocity and angular velocity is infinity
        upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(55)]
        lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(55)]
        
        # Convert continous features into discrete values
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        discrete_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        discrete_obs = [min(self.buckets[i] - 1, max(0, discrete_obs[i])) for i in range(len(obs))]
        return tuple(discrete_obs)

    def choose_action(self, state):
        # Follow an epsilon-greedy policy to choose the next action
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])
        
    def update_q_table(self, state_old, action, reward, state_new):
        # Bellman equation - update the Q-table
        self.q_table[state_old][action] += self.lr * (reward + self.gamma * np.max(self.q_table[state_new]) - self.q_table[state_old][action])


    def update_epsilon(self):
        self.epsilon = max(0.1, self.epsilon * self.epsilon_decay)

    def update_learning_rate(self):
        self.lr = max(0.1, self.lr * self.lr_decay)
    
    def is_solved(self, episode, rewards):
        if len(rewards) > 50:
            if all(r >= 400 for r in rewards[-50:]):
                print("Solved at episode {}.".format(episode - 50))
                return True
            else:
            	return False

    def plot_rewards_history(self, rewards):
    	# Plot rewards over episodes
        episodes = list(range(1, len(rewards)+1))
        plt.plot(episodes, rewards)
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.show()
        return

    def train_agent(self):
        rewards = list()

        for episode in range(self.episodes):
            current_state = self.discretize_observation(self.env.reset())
            done = False
            step = 0

            while step < MAX_STEPS:
                self.env.render()
                action = self.choose_action(current_state)
                obs, reward, done, _ = self.env.step(action)

                new_state = self.discretize_observation(obs)
                self.update_q_table(current_state, action, reward, new_state)
                current_state = new_state

                if done:
                    break
                step += 1

            rewards.append(step)
            if self.is_solved(episode, rewards):
            	return rewards

            self.update_learning_rate()
            self.update_epsilon()

        return rewards

if __name__ == "__main__":
    solver = QLearningAgent()
    rewards = solver.train_agent()
    solver.plot_rewards_history(rewards)
