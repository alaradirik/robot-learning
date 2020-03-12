import math

import gym
import numpy as np
from matplotlib import pyplot as plt


""" 
CartPole-v1 is considered "solved" when the agent obtains a reward of at
least 400 over 50 consecutive episodes.
"""

MAX_STEPS = 500

class DeepQAgent():
    def __init__(self, episodes=2000, lr=0.01, epsilon=1.0, epsilon_min=0.01, gamma=0.9):
        """
        Construct a new Deep Q-learning agent.

        :param episodes: Number of training episodes 
        :param lr: Learning rate
        :param epsilon: Exploration rate
        :param epsilon: Minimum decayed exploration rate
        :param gamma: Reward discount rate

        :return: returns nothing
        """

        self.episodes = episodes
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.lr = lr
        self.gamma = gamma
        
        # Load CartPole V1 environment
        self.env = gym.make('CartPole-v1')
        self.env._max_episode_steps = MAX_STEPS
        
        # initialize network variables
        self.input = self.env.observation_space.shape[0] + 5
        self.w1 = np.random.uniform(0,1,(self.input, 1))

    def take_action(self, propagations):
    	# # Follow an epsilon-greedy policy to choose the next action
        if np.random.random() < self.epsilon: 
            return self.env.action_space.sample()
        else:
            return np.argmax(propagations) 
     
    def update_epsilon(self, t):
        self.epsilon = max(self.epsilon_min, self.epsilon * 0.99)


    def is_solved(self, episode, rewards):
    	# Check if game is solved
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
    

    def compute_state_action_vector(self, observation, action):
        f0_vector = np.append([1], observation)
        f0_vector = np.append(f0_vector, [0, 0, 0, 0])

        f1_vector = np.append([1], [0, 0, 0, 0])
        f1_vector = np.append(f1_vector, observation)

		# Return the corresponding vector based on action: left or right
        if action == "predict":
            return f0_vector, f1_vector
        if action == 0:
            return f0_vector
        if action == 1:
            return f1_vector
  
    def forward_pass(self, f_vector):
    	# Outputs the Q value of the given vector
        return np.dot(self.w1.T, f_vector)
    
    def train_network(self, current_state, next_state, action, reward, terminate=False):
    	# Train the feedforward neural network
        forward_vector = self.compute_state_action_vector(current_state, action)
        q_value = self.forward_pass(forward_vector)

        f0_vector, f1_vector = self.compute_state_action_vector(next_state, action="predict")
        props = [self.forward_pass(f0_vector), self.forward_pass(f1_vector)]
        q_max = max(props)
         
        # Return reward if the episode is terminated
        if terminate:
            target = reward
        else:
            target = reward + self.gamma * q_max
          
        error = q_value - target
        error = np.clip(error, -30, 30)

        self.w1 += self.lr * error * forward_vector.reshape((-1, 1))
        self.w1 = np.clip(self.w1, -150, 150)
        return error, props
    
    def train_agent(self):
        scores = []
        error = []

        for episode in range(self.episodes):
            current_state = self.env.reset()
            done = False
            step = 0
              
            f0_vector, f1_vector = self.compute_state_action_vector(current_state, action="predict")
            predicted_q_values = [self.forward_pass(f0_vector), self.forward_pass(f1_vector)]
            action = self.take_action(predicted_q_values)
         
          
            while step < MAX_STEPS:
           
                new_state, reward, done, info = self.env.step(action)

                if done:
                    break

                error_t, predicted_q_values = self.train_network(current_state, new_state, action, reward)
                action = self.take_action(predicted_q_values)
                current_state = new_state

                step += 1

          
            scores.append(step) 
            error.append(error_t)
            if self.is_solved(episode, scores):
                return scores, error_t

            self.update_epsilon(episode)
        return scores, error


solver = DeepQAgent()
rewards, error = solver.train_agent()
solver.plot_rewards_history(rewards)


