import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from algorithm.model import BaseModel

class REINFORCENetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(REINFORCENetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.LogSoftmax(dim=-1)
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
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.model_path = model_path
        
        # Environment info
        self.state_dim = self.num_states
        self.action_dim = len(self.action_space)
        
        # Policy Network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = REINFORCENetwork(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Logging setup
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.log_path = os.path.join(os.path.dirname(self.model_path), "training_log.txt")
        with open(self.log_path, "w") as f:
            f.write("REINFORCE Training Log\n")

    def predict(self, state, training=False):
        state_idx = self.state_to_index(state)
        state_tensor = torch.eye(self.num_states)[state_idx].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            log_probs = self.policy_net(state_tensor)
            probs = torch.exp(log_probs).squeeze(0) 
            
        if training:
            action_idx = torch.multinomial(probs, 1).item()
        else:
            action_idx = torch.argmax(probs).item()
            
        return self.action_space[action_idx]

    def train(self):
        print("REINFORCE Training...")
        best_reward = -float('inf')
        
        for episode in range(self.iterations):
            state = self.start_state
            episode_states = []
            episode_actions = []
            episode_rewards = []
            
            # get an episode
            while state != self.target_state:
                action = self.predict(state, training=True)
                next_state, reward = self.env.get_next_state_and_reward(state, action)
                
                # save (state, action, reward)
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)
                
                state = next_state
            
            # calculate q(s,a) of every pair (s,a) in the episode
            returns = []
            R = 0
            for r in reversed(episode_rewards):
                R = r + self.gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
            
            # Compute policy gradient
            policy_losses = []
            for s, a, R in zip(episode_states, episode_actions, returns):
                state_idx = self.state_to_index(s)
                state_tensor = torch.eye(self.num_states)[state_idx].unsqueeze(0).to(self.device)
                log_probs = self.policy_net(state_tensor)
                action_idx = self.action_space.index(a)
                policy_loss = -log_probs[0, action_idx] * R
                policy_losses.append(policy_loss)

            self.optimizer.zero_grad()
            loss = torch.stack(policy_losses).sum()
            loss.backward()
            self.optimizer.step()
            
            total_reward = sum(episode_rewards)
            with open(self.log_path, "a") as f:
                f.write(f"Episode {episode+1}, Reward: {total_reward:.2f}, Loss: {loss.item():.4f}\n")
            
            # Save latest model
            print(f"episode {episode + 1}")
            latest_path = self.model_path.replace(".pth", "_latest.pth")
            self.save(latest_path)
            
            if total_reward > best_reward:
                best_reward = total_reward
                self.save(self.model_path)
                print(f"[Best] Episode {episode+1}, Reward: {total_reward:.2f}")

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