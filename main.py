import gym
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import namedtuple, deque
from model.dqnAgent import DQNAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Set up the environment
env = gym.make("CartPole-v1")

# Define training parameters
num_episodes = 250
max_steps_per_episode = 200
epsilon_start = 1.0
epsilon_end = 0.2
epsilon_decay_rate = 0.99
gamma = 0.9
lr = 0.0025
buffer_size = 10000
buffer = deque(maxlen=buffer_size)
batch_size = 128
update_frequency = 10


# Initialize the DQNAgent
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n


new_agent = DQNAgent(input_dim, output_dim, seed=170715, lr = lr)

# Training loop
for episode in range(num_episodes):
    # Reset the environment
    state = env.reset()
    epsilon = max(epsilon_end, epsilon_start * (epsilon_decay_rate ** episode))

    # Run one episode
    for step in range(max_steps_per_episode):
        # Choose and perform an action
        action = new_agent.act(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        
        buffer.append((state, action, reward, next_state, done))
        
        if len(buffer) >= batch_size:
            batch = random.sample(buffer, batch_size)
            # Update the agent's knowledge
            new_agent.learn(batch, gamma)
        
        state = next_state
        # Check if the episode has ended
        if done:
            break
    
    if (episode + 1) % update_frequency == 0:
        print(f"Episode {episode + 1}: Finished training")

# Evaluate the agent's performance
test_episodes = 100
episode_rewards = []

for episode in range(test_episodes):
    state = env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        action = new_agent.act(state, eps=0.)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        state = next_state
        
    episode_rewards.append(episode_reward)

average_reward = sum(episode_rewards) / test_episodes
print(f"Average reward over {test_episodes} test episodes: {average_reward:.2f}")

# Visualize the agent's performance
import time

state = env.reset()
done = False

while not done:
    env.render()
    action = new_agent.act(state, eps=0.)
    next_state, reward, done, _ = env.step(action)
    state = next_state
    time.sleep(0.1)  # Add a delay to make the visualization easier to follow

env.close()