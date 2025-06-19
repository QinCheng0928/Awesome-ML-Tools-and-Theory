import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from algorithm.model import BaseModel

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        
    def forward(self, x):
        return self.net(x)

class DQN(BaseModel):
    def __init__(self, 
                 env, 
                 gamma=0.99, 
                 iterations=1000, 
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.995,
                 batch_size=64,
                 buffer_size=10000,
                 target_update_freq=10,
                 learning_rate=0.001, 
                 model_path='../log/DQN/DQN_model_v0.pth'):
        super().__init__(env, gamma)
        
        # Hyperparameters
        self.iterations = iterations
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.target_update_freq = target_update_freq
        self.learning_rate = learning_rate
        self.model_path = model_path
        
        # Environment info
        self.state_dim = self.num_states
        self.action_dim = len(self.action_space)
        
        # Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        # update target network using policy network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimization
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Replay buffer
        self.memory = deque(maxlen=self.buffer_size)
        
        # Logging setup
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.log_path = os.path.join(os.path.dirname(self.model_path), "training_log.txt")
        with open(self.log_path, "w") as f:
            f.write("DQN Training Log\n")

    def predict(self, state, training=False):
        if training and random.random() < self.epsilon:
            return random.choice(self.action_space)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        action_idx = torch.argmax(q_values, dim=1).item()
        return self.action_space[action_idx]

    def store_transition(self, state, action, reward, next_state, done):
        action_idx = self.action_space.index(action)
        self.memory.append((state, action_idx, reward, next_state, done))

    def train(self):
        print("DQN Training...")
        best_loss = float('inf')
        
        for episode in range(self.iterations):
            state = self.start_state
            total_reward = 0
            total_loss = 0
            step_count = 0
            
            while state != self.target_state:
                # Select and execute action
                action = self.predict(state, training=True)
                next_state, reward = self.env.get_next_state_and_reward(state, action)
                done = (next_state == self.target_state)
                
                # Store transition
                self.store_transition(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                step_count += 1
                
                # Train if enough samples in buffer
                if len(self.memory) >= self.batch_size:
                    loss = self._update_model()
                    total_loss += loss
            
            # Decay epsilon: Gradually reduce exploration
            self.epsilon = max(self.epsilon_end, 
                              self.epsilon * self.epsilon_decay)
            
            # Update target network
            if episode % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # Logging
            avg_loss = total_loss / step_count if step_count > 0 else 0
            with open(self.log_path, "a") as f:
                f.write(f"Episode {episode+1}, Reward: {total_reward:.2f}, Loss: {avg_loss:.4f}, Epsilon: {self.epsilon:.3f}\n")
            
            # Save models
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save(self.model_path)
                print(f"[Best] Episode {episode+1}, Reward: {total_reward:.2f}, Loss: {avg_loss:.4f}")
            
            # Save latest model
            latest_path = self.model_path.replace(".pth", "_latest.pth")
            self.save(latest_path)
            
            print(f"Episode {episode+1}, Reward: {total_reward:.2f}, Loss: {avg_loss:.4f}, Epsilon: {self.epsilon:.3f}")

        print("Training completed.")

    def _update_model(self):
        # Sample batch from replay buffer
        batch = random.sample(self.memory, self.batch_size)
        # Every various is a list
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert states to indices and then to one-hot encoding
        states_indices = torch.tensor(
            [self.state_to_index(s) for s in states],
            dtype=torch.long,
            device=self.device
        )
        next_states_indices = torch.tensor(
            [self.state_to_index(s) for s in next_states],
            dtype=torch.long,
            device=self.device
        )

        # Create one-hot encodings and Convert to tensors
        states = torch.eye(self.num_states, device=self.device)[states_indices].float()
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.eye(self.num_states, device=self.device)[next_states_indices].float()
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions)
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Compute loss and update
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def save(self, path=None):
        if path is None:
            path = self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        print(f"Model saved to {path}")

    def load(self, path=None):
        if path is None:
            path = self.model_path
        assert os.path.exists(path), f"Model file {path} not found."
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        print(f"Model loaded from {path}")