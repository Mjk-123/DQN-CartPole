import gym
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import namedtuple, deque
import wandb
import os
import time

from model.dqnAgent import DQNAgent
from model.pgAgent import PolicyGradientAgent

# -------------------------------
# 1. Initialize wandb
# -------------------------------
wandb.init(
    project="DQN-CartPole",
    name="policy-gradient_learn",
    config={
        "num_episodes": 250,
        "max_steps": 200,
        "epsilon_start": 1.0,
        "epsilon_end": 0.2,
        "epsilon_decay": 0.99,
        "gamma": 0.9,
        "lr": 0.0025,
        "batch_size": 128,
    }
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Checkpoint directory
ckpt_dir = "./checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)

# Initialize the DQNAgent
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

# new_agent = DQNAgent(input_dim, output_dim, seed=170715, lr=lr)
agent = PolicyGradientAgent(
    state_size=4,
    action_size=2,
    seed=170715,
    lr=1e-3,
    gamma=0.99
)

env = gym.make("CartPole-v1")

agent = PolicyGradientAgent(
    state_size=4,
    action_size=2,
    seed=170715,
    lr=1e-3,
    gamma=0.99
)

num_episodes = 1000

# Training loop

for ep in range(num_episodes):
    state = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.store_reward(reward)

        state = next_state
        episode_reward += reward

    loss = agent.update()

    # -------------------------------
    # WandB Logging
    # -------------------------------
    wandb.log({
        "episode": ep,
        "reward": episode_reward,
        "loss": loss
    })

    # -------------------------------
    # Save checkpoint every 50 episodes
    # -------------------------------
    if (ep + 1) % 50 == 0:
        ckpt_path = f"{ckpt_dir}/pg_episode_{ep+1}.pth"
        torch.save({
            "policy_state_dict": agent.policy.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict()
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    if ep % 10 == 0:
        print(f"Episode {ep}: reward={episode_reward}, loss={loss:.4f}")


# ------------------------------------------------
# Save FINAL checkpoint at the end of training
# ------------------------------------------------

final_ckpt_path = f"{ckpt_dir}/pg_final.pth"
torch.save({
    "policy_state_dict": agent.policy.state_dict(),
    "optimizer_state_dict": agent.optimizer.state_dict()
}, final_ckpt_path)
print(f"Final checkpoint saved to: {final_ckpt_path}")

# ----------------------------------
# Evaluation (deterministic policy)
# ----------------------------------

test_episodes = 100
episode_rewards = []

for ep in range(test_episodes):
    state = env.reset()
    done = False
    episode_reward = 0

    while not done:
        # evaluation은 deterministic
        action = agent.act_deterministic(state)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        state = next_state

    episode_rewards.append(episode_reward)

avg_reward = sum(episode_rewards) / test_episodes
print(f"Average reward over {test_episodes} episodes = {avg_reward:.2f}")

# ----------------------------------
# Visualization
# ----------------------------------

state = env.reset()
done = False

while not done:
    env.render()
    action = agent.act_deterministic(state)
    next_state, reward, done, _ = env.step(action)
    state = next_state
    time.sleep(0.05)

env.close()
