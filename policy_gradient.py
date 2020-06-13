import gym
import numpy as np
from matplotlib import pyplot as plt


class Policy:
    def __init__(self, n_states, n_actions, n_hidden):
        self.model = {
            'W1': np.random.randn(n_states, n_hidden) / np.sqrt(n_states),
            'W2': np.random.randn(n_hidden, 1) / np.sqrt(n_hidden),
            'b1': np.zeros((n_hidden,)),
            'b2': np.zeros((1,))
        }
        # Batch gradient buffer
        self.grad_buffer = {k: np.zeros_like(v) for k, v in self.model.items()}
        # RMSProp memory
        self.rmsprop_cache = {k: np.zeros_like(v) for k, v in self.model.items()}
        self.x_cache, self.a1_cache, self.rewards, self.dlogprobs = [], [], [], []
  
    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))

    def relu(self, z): 
        return np.maximum(0, z)

    def forward(self, x):
        # Forward propogation. Return the probabilities of actions and hidden state
        z1 = np.dot(x, self.model['W1']) + self.model['b1']
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.model['W2']) + self.model['b2']
        a_prob0 = self.sigmoid(z2)
        a_prob = np.array([a_prob0, 1-a_prob0]).flatten()
        return a_prob, a1   

    def backward(self, x_cache, a1_cache, dZ2):
        # Backward propogation. Return the gradients of parameters
        dW2 = np.dot(a1_cache.T, dZ2)
        db2 = np.sum(dZ2, axis=0,)
        dZ1 = np.dot(self.model['W2'], dZ2.T)
        dZ1[a1_cache.T <= 0] = 0
        dW1 = np.dot(x_cache.T, dZ1.T)
        db1 = np.sum(dZ1, axis=1,)
        return {'W1': dW1, 'W2': dW2, 'b1': db1, 'b2': db2}


class Reinforce:
    def __init__(self, seed, n_states, n_actions, n_hidden, lr=3e-3, gamma=0.99):
        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.model = Policy(n_states, n_actions, n_hidden)

    def add_step_reward(self, reward):
        self.model.rewards.append(reward)

    def get_rewards_sum(self):
        return np.sum(self.model.rewards)

    def discount_rewards(self, model_rewards):
        # Return discounted rewards
        running_add = 0
        discounted_rewards = []

        for r in model_rewards[::-1]:
            running_add = r + self.gamma * running_add
            discounted_rewards.insert(0, running_add)
        return (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-9)

    def choose_action(self, state):
        # Choose action based on policy
        aprob, a1 = self.model.forward(state)
        action = np.random.choice(self.n_actions, p=aprob)
        y = np.zeros_like(aprob)
        y[action] = 1
 
        self.model.x_cache.append(state)
        # Update hidden state cache
        self.model.a1_cache.append(a1)
        self.model.dlogprobs.append(y[0] - aprob[0])
        return action

    def update_policy(self):
        x_cache = np.vstack(self.model.x_cache)
        a1_cache = np.vstack(self.model.a1_cache)
        discounted_rewards = np.array(self.discount_rewards(self.model.rewards)).reshape(-1, 1)
        dlogprobs_advantage = np.vstack(self.model.dlogprobs) * discounted_rewards
        
        # Compute gradients and update parameters
        grad = self.model.backward(x_cache, a1_cache, dlogprobs_advantage)
        for k in self.model.model:
            self.model.grad_buffer[k] += grad[k]
        
        for k, v in self.model.model.items():
            g = self.model.grad_buffer[k]  
            self.model.rmsprop_cache[k] = self.gamma * self.model.rmsprop_cache[k] + (1 - self.gamma) * g ** 2
            self.model.model[k] += self.lr * g / (np.sqrt(self.model.rmsprop_cache[k]) + 1e-5)
            # Reset the batch gradient buffer
            self.model.grad_buffer[k] = np.zeros_like(v)  

        self.model.rewards, self.model.x_cache, self.model.a1_cache, self.model.dlogprobs = [], [], [], []


def is_solved(episode, all_rewards):
    # Return true if the avg reward over 10 cons. episodes is over 450 
    if len(all_rewards) > 10:
        if np.mean(np.array(all_rewards)[-10:]) >= 450:
            return True
        else:
            return False

def plot_rewards_history(rewards):
    # Plot rewards over episodes
    episodes = list(range(1, len(rewards)+1))
    plt.plot(episodes, rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.show()
    return    

def main():
    SEED = 17
    MAX_EPISODES = 2000

    env = gym.make('CartPole-v1')
    env.seed(SEED)
    np.random.seed(SEED)
    
    # Total reward of each episode
    all_rewards = []

    agent = Reinforce(seed=SEED, n_states=env.observation_space.shape[0], n_actions=env.action_space.n, n_hidden=4)

    for i in range(MAX_EPISODES):
        # Reset state
        state = env.reset()
        
        # Default max episode steps value is 500
        for j in range(env._max_episode_steps):
            action = agent.choose_action(state)
            state, reward, done, _ = env.step(action)
            agent.add_step_reward(reward)
            env.render()

            if done:
                break
    
        episode_reward = agent.get_rewards_sum()
        all_rewards.append(episode_reward)
        print('Episode {}\tLength: {:5d}\tEpisode Reward: {:.2f}'.format(i, (j + 1), episode_reward))
        agent.update_policy()

        if is_solved(i, all_rewards):
            break
    
    plot_rewards_history(all_rewards)
    env.close()


if __name__ == '__main__':
    main()

