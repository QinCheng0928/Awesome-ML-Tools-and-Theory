import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
from algorithm.model import BaseModel

class Actor(nn.Module):
    """Actor Network"""
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    """Critic Network"""
    def __init__(self, state_dim, hidden_dim=32):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, log_prob):
        self.buffer.append((state, action, reward, next_state, done, log_prob))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, old_log_probs = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.long),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.long),
            torch.tensor(dones, dtype=torch.float32),
            torch.tensor(old_log_probs, dtype=torch.float32)
        )
    
    def __len__(self):
        return len(self.buffer)

class PPO(BaseModel):
    """PPO algorithm"""
    def __init__(self, 
                 env, 
                 gamma=0.99, 
                 iterations=1000, 
                 learning_rate=0.001,
                 clip_epsilon=0.2,
                 ppo_epochs=4,
                 batch_size=64,
                 buffer_size=10000,
                 model_path='../log/PPO/PPO_model_v0.pth'):
        super().__init__(env, gamma)
        
        # Hyperparameters
        self.iterations = round(iterations)
        self.learning_rate = learning_rate
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.model_path = model_path
        
        # Environment info
        self.state_dim = self.num_states
        self.action_dim = len(self.action_space)
        
        # Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(self.state_dim, self.action_dim).to(self.device)
        self.critic = Critic(self.state_dim).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        
        # Replay Buffer
        self.buffer = ReplayBuffer(buffer_size)

    def predict(self, state, training=False):
        state_idx = self.state_to_index(state)
        state_tensor = torch.eye(self.num_states)[state_idx].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            probs = self.actor(state_tensor)
            
        if training:
            action_idx = torch.multinomial(probs, 1).item()
        else:
            action_idx = torch.argmax(probs).item()
            
        return self.action_space[action_idx]

    def train(self):
        print("Off-Policy PPO Training...")
        
        for i in range(self.iterations):
            print(f"Iteration {i} training...")
            
            # get data
            state = self.start_state
            while state != self.target_state:
                state_idx = self.state_to_index(state)
                state_tensor = torch.eye(self.num_states)[state_idx].unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    action_probs = self.actor(state_tensor)
                    value = self.critic(state_tensor)
                
                action_idx = torch.multinomial(action_probs, 1).item()
                action = self.action_space[action_idx]
                next_state, reward = self.env.get_next_state_and_reward(state, action)
                done = (next_state == self.target_state)
                
                log_prob = torch.log(action_probs.squeeze(0)[action_idx])
                
                self.buffer.push(state_idx, action_idx, reward, self.state_to_index(next_state), done, log_prob.item())
                state = next_state
            
            if len(self.buffer) < self.batch_size:
                continue
                
            for _ in range(self.ppo_epochs):
                states, actions, rewards, next_states, dones, old_log_probs = self.buffer.sample(self.batch_size)
                states = states.to(self.device)
                actions = actions.to(self.device)
                rewards = rewards.to(self.device)
                next_states = next_states.to(self.device)
                dones = dones.to(self.device)
                old_log_probs = old_log_probs.to(self.device)
                
                state_tensors = torch.eye(self.num_states).to(self.device)[states]
                next_state_tensors = torch.eye(self.num_states).to(self.device)[next_states]
                
                # Importance sampling ratio
                new_action_probs = self.actor(state_tensors)
                new_log_probs = torch.log(new_action_probs.gather(1, actions.unsqueeze(1))).squeeze()
                ratios = torch.exp(new_log_probs - old_log_probs)
                
                print("probs : ", new_action_probs[20])
                
                # Compute advantages
                new_values = self.critic(state_tensors)
                with torch.no_grad():
                    next_values = self.critic(next_state_tensors)
                    targets = rewards + (1 - dones) * self.gamma * next_values.squeeze()
                    advantages = targets - new_values.squeeze()
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Actor loss
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Critic loss
                value_loss = F.mse_loss(new_values.squeeze(), targets)
                
                # update
                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                self.actor_optimizer.step()
                

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                self.critic_optimizer.step()

        print("Training completed.")

    def save(self, path=None):
        """保存模型（同时保存 Actor 和 Critic）"""
        if path is None:
            path = self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")

    def load(self, path=None):
        """加载模型（同时加载 Actor 和 Critic）"""
        if path is None:
            path = self.model_path
        assert os.path.exists(path), f"Model file {path} not found."
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        print(f"Model loaded from {path}")