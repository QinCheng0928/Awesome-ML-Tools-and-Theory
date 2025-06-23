import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from algorithm.model import BaseModel

class REINFORCENetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super(REINFORCENetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        return self.net(x)
    
class REINFORCE(BaseModel):
    def __init__(self, 
                 env, 
                 gamma=0.99, 
                 iterations=1000, 
                 learning_rate=0.001,
                 model_path='../log/REINFORCE/REINFORCE_model_v0.pth'):
        super().__init__(env, gamma)
        
        # Hyperparameters
        self.iterations = round(iterations)
        self.learning_rate = learning_rate
        self.model_path = model_path
        
        # Environment info
        self.state_dim = self.num_states
        self.action_dim = len(self.action_space)
        
        # Policy Network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = REINFORCENetwork(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

    def predict(self, state, training=False):
        state_idx = self.state_to_index(state)
        state_tensor = torch.eye(self.num_states)[state_idx].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            probs = self.policy_net(state_tensor)
            
        if training:
            action_idx = torch.multinomial(probs, 1).item()
        else:
            action_idx = torch.argmax(probs).item()
            
        return self.action_space[action_idx]

    def train(self):
        print("REINFORCE Training...")
        
        for i in range(self.iterations):
            print(f"iterations { i } training...")

            
            state = self.start_state
            episode_states = []
            episode_actions = []
            episode_rewards = []
            
            while state != self.target_state:
                action = self.predict(state, training=True)
                next_state, reward = self.env.get_next_state_and_reward(state, action)
                
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)
                
                state = next_state         
            
            returns = []
            R = 0
            for r in reversed(episode_rewards):
                R = r + self.gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
            
            state_indices = [self.state_to_index(s) for s in episode_states]
            states_tensor = torch.eye(self.num_states)[state_indices].to(self.device)
            action_indices = torch.tensor([self.action_space.index(a) for a in episode_actions], 
                                        dtype=torch.long, device=self.device)
            
            probs = self.policy_net(states_tensor)[range(len(action_indices)), action_indices]
            log_probs = torch.log(probs)
            policy_loss = -(log_probs * returns).sum()
            
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

        print("Training completed.")

    def save(self, path=None):
        if path is None:
            path = self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")

    def load(self, path=None):
        if path is None:
            path = self.model_path
        assert os.path.exists(path), f"Model file {path} not found."
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Model loaded from {path}")