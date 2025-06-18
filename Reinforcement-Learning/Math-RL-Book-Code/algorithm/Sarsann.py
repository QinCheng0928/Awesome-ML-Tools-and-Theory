import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from algorithm.model import BaseModel

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class SarsaNN(BaseModel):
    def __init__(self, 
                 env, 
                 gamma=0.9, 
                 iterations=1000, 
                 epsilon=0.1, 
                 alpha=0.01, 
                 model_path='../log/Sarsa_nn/Sarsa_nn_model_v0.pth'):
        super().__init__(env, gamma)
        self.iterations = iterations
        self.epsilon = epsilon
        self.alpha = alpha
        self.model_path = model_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = QNetwork(state_dim=self.num_states, action_dim=len(self.action_space)).to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss()

    def select_action(self, q_values, i):
        if np.random.rand() < max(0.01, 1.0 - i / self.iterations):
            return np.random.randint(len(self.action_space))
        else:
            return torch.argmax(q_values).item()

    def predict(self, state):
        state_idx = self.state_to_index(state)
        state_tensor = torch.eye(self.num_states)[state_idx].unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        action_idx = torch.argmax(q_values, dim=1).item()
        action = self.action_space[action_idx]
        return action

    def train(self):
        print("Sarsa with value function approximation Training...")
        
        best_loss = float('inf')
        log_path = os.path.join(os.path.dirname(self.model_path), "training_log.txt")
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        for i in range(self.iterations):
            state = self.start_state
            state_idx = self.state_to_index(state)

            # eye(n) create an identity matrix of size n
            # each row represents a one-hot vector
            # unsqueeze insert a dimension at the 0th dimension to change the shape from [num_states] to [1, num_states].
            state_tensor = torch.eye(self.num_states)[state_idx].unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
            action_idx = self.select_action(q_values, i)
            action = self.action_space[action_idx]

            total_loss = 0.0
            
            while state != self.target_state:
                # here, we have r_t+1
                next_state, reward = self.env.get_next_state_and_reward(state, action)
                next_state_idx = self.state_to_index(next_state)

                state_tensor = torch.eye(self.num_states)[state_idx].unsqueeze(0).to(self.device)
                next_state_tensor = torch.eye(self.num_states)[next_state_idx].unsqueeze(0).to(self.device)
                reward_tensor = torch.tensor(reward, dtype=torch.float32).to(self.device)

                # Predict q(s_t,a_t)
                q_values = self.q_net(state_tensor)
                q_value = q_values[0, action_idx]

                # Predict q(s_t+1,a_t+1)
                with torch.no_grad():
                    next_q_values = self.q_net(next_state_tensor)
                next_action_idx = self.select_action(next_q_values, i)
                next_q_value = next_q_values[0, next_action_idx]

                # Sarsa target
                target = reward_tensor + self.gamma * next_q_value

                # Backpropagation
                loss = self.loss_fn(q_value, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                # Move to next state
                state = next_state
                state_idx = next_state_idx
                action_idx = next_action_idx
                action = self.action_space[action_idx]
                
            # Logging loss
            with open(log_path, "a") as f:
                f.write(f"Iter {i+1}, Loss: {total_loss:.4f}\n")

            # Save best model
            if total_loss < best_loss:
                best_loss = total_loss
                self.save(self.model_path)
                print(f"[Best] New best model saved with loss {best_loss:.4f}")

            # Save latest model for resume
            latest_path = self.model_path.replace(".pth", "_latest.pth")
            self.save(latest_path)
            
        print("Training completed.")


    def save(self, path=None):
        if path is None:
            path = self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.q_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path=None):
        if path is None:
            path = self.model_path
        assert os.path.exists(path), f"Model file {path} not found."
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")
