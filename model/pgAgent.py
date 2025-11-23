import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import namedtuple, deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return F.softmax(logits, dim=-1)


class PolicyGradientAgent:
    def __init__(self, state_size, action_size, seed=0, lr=1e-3, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        
        self.policy = PolicyNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr)

        # trajectory 저장용 리스트
        self.log_probs = []
        self.rewards = []

    def act(self, state):
        """ 정책 확률에서 행동 샘플링 """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.policy(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        # 저장해두기 (나중에 update에 사용)
        self.log_probs.append(dist.log_prob(action))

        return action.item()

    def update(self):
        """ Episode 기반 REINFORCE 업데이트 """

        # 1. discounted return G_t 계산
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # 안정성 증가

        # 2. policy gradient loss
        loss = 0
        for logp, Gt in zip(self.log_probs, returns):
            loss += -logp * Gt

        # 3. optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 4. episode 저장기 초기화
        self.log_probs = []
        self.rewards = []

        return loss.item()

    def store_reward(self, reward):
        self.rewards.append(reward)

    def act_deterministic(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.policy(state)
        action = torch.argmax(probs, dim=1)
        return action.item()

