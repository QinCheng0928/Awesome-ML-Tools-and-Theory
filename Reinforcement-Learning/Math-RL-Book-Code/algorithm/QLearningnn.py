import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from algorithm.model import BaseModel

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.linear = nn.Linear(state_dim, action_dim)

    def forward(self, x):
        return self.linear(x)

class QLearningNN(BaseModel):
    def __init__(self, 
                 env, 
                 gamma=0.9, 
                 iterations=1000, 
                 epsilon=0.1, 
                 alpha=0.01, 
                 model_path='../log/QLearning_nn/QLearning_nn_model_v0.pth'):
        super().__init__(env, gamma)
        self.iterations = iterations
        self.epsilon = epsilon
        self.alpha = alpha
        self.model_path = model_path

        self.num_actions = len(self.action_space)
        self.policy = np.ones((self.num_states, len(self.action_space))) / len(self.action_space)  # policy
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = QNetwork(state_dim=self.num_states, action_dim=len(self.action_space)).to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss()

    def predict(self, state):
        state_idx = self.state_to_index(state)
        state_tensor = torch.eye(self.num_states)[state_idx].unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        action_idx = torch.argmax(q_values, dim=1).item()
        action = self.action_space[action_idx]
        return action


    def train(self):
        print("Q-Learning with value function approximation Training...")
        best_loss = float('inf')
        log_path = os.path.join(os.path.dirname(self.model_path), "training_log.txt")
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(log_path, "w") as f:
            f.write("Q-Learning Training Log\n")

        for i in range(self.iterations):
            print(f"iterations { i } starts training...")
            state_t = self.start_state
            state_t_idx = self.state_to_index(state_t)
            action_prob = self.policy[state_t_idx]
            action_t_idx = np.random.choice(len(action_prob), p=action_prob)
            action_t = self.action_space[action_t_idx]

            total_loss = 0.0

            while state_t != self.target_state:
                state_t1, reward_t1 = self.env.get_next_state_and_reward(state_t, action_t)
                state_t1_idx = self.state_to_index(state_t1)
                action_prob1 = self.policy[state_t1_idx]
                action_t1_idx = np.random.choice(len(action_prob1),p=action_prob1)
                action_t1 = self.action_space[action_t1_idx]

                state_t_tensor = torch.eye(self.num_states)[state_t_idx].unsqueeze(0).to(self.device)
                state_t1_tensor = torch.eye(self.num_states)[state_t1_idx].unsqueeze(0).to(self.device)

                # q(s_t, a_t)
                q_values_t = self.q_net(state_t_tensor)
                q_pred = q_values_t[0, action_t_idx]

                # q(s_t+1, a_t+1)
                with torch.no_grad():
                    q_values_t1 = self.q_net(state_t1_tensor)
                    q_next = torch.max(q_values_t1[0])

                # TD target
                target = reward_t1 + self.gamma * q_next

                # Update network
                loss = self.loss_fn(q_pred, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                # updata policy
                for s_idx in range(self.num_states):
                    onehot_state = torch.eye(self.num_states)[s_idx].unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        best_action_idx = torch.argmax(self.q_net(onehot_state)).item()
                    self.policy[s_idx, :] = self.epsilon / self.num_actions
                    self.policy[s_idx, best_action_idx] = 1 - self.epsilon + self.epsilon / self.num_actions
                    
                # Update state and actions
                state_t = state_t1
                state_t_idx = state_t1_idx
                action_t_idx = action_t1_idx
                action_t = action_t1
                

            # logging
            with open(log_path, "a") as f:
                f.write(f"Iter {i+1}, Loss: {total_loss:.4f}\n")

            # Save the best model
            if total_loss < best_loss:
                best_loss = total_loss
                self.save(self.model_path)
                print(f"[Best] New best model saved with loss {best_loss:.4f}")

            # Save latest model
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
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.q_net.load_state_dict(state_dict)
        print(f"Model loaded from {path}")
