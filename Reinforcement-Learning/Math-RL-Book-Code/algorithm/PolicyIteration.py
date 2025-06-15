import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)
from algorithm.model import BaseModel
import numpy as np
import torch
print("cuda is avaliable", torch.cuda.is_available())


class PolicyIteration(BaseModel):
    def __init__(self, 
                 env,
                 thousands=1e-5,
                 gamma=0.9,
                 ):
        super().__init__(env, thousands, gamma)
        
        self.v = np.zeros(self.num_states)                                 # state values
        self.q = np.zeros((self.num_states, len(self.action_space)))       # action values
        self.policy = np.zeros((self.num_states, len(self.action_space)))  # policy
    
    # (x, y) to state index
    def state_to_index(self, state):
        x, y = state
        return y * self.env_size[0] + x
    
    # state index to (x, y)
    def index_to_state(self, index):
        x = index % self.env_size[0]
        y = index // self.env_size[0]
        return (x, y)
    
    # Get the current action based on current state and policy
    def predict(self, state):
        state_index = self.state_to_index(state)
        action_index = np.argmax(self.policy[state_index])
        return self.action_space[action_index]
    
    def train(self):
        iteration = 0
        while True:
            iteration += 1
            print(f"Iteration {iteration}: Policy Evaluation...")

            # Policy Evaluation
            while True:
                delta = 0
                last_v = self.v.copy()
                for state_index in range(self.num_states):
                    v_new = 0
                    for action_index, action in enumerate(self.action_space):
                        next_state, reward = self.env.get_next_state_and_reward(self.index_to_state(state_index), action)
                        next_state_index = self.state_to_index(next_state)
                        v_new += self.policy[state_index][action_index] * (reward + self.gamma * last_v[next_state_index])
                    delta = max(delta, abs(v_new - self.v[state_index]))                
                    self.v[state_index] = v_new

                if delta < self.thousands:
                    break

            # Policy Improvement
            policy_stable = True
            for state_index in range(self.num_states):
                old_action = np.argmax(self.policy[state_index])
                for action_index, action in enumerate(self.action_space):
                    next_state, reward = self.env.get_next_state_and_reward(self.index_to_state(state_index), action)
                    next_state_index = self.state_to_index(next_state)
                    self.q[state_index][action_index] = reward + self.gamma * self.v[next_state_index]
                best_action = np.argmax(self.q[state_index])
                if best_action != old_action:
                    policy_stable = False

                self.policy[state_index] = np.zeros(len(self.action_space))
                self.policy[state_index][best_action] = 1.0

            # Check if the policy is stable
            # This is different from judging whether v converges in the book. Here, it is to judge whether the strategy converges.
            if policy_stable:
                print(f"Policy converged after {iteration} iterations.")
                break

        print("Policy Iteration completed.")


     
    # load the policy, action values and state values from a file
    def load(self, path):
        self.model_path = path
        assert os.path.exists(self.model_path), f"Model file '{self.model_path}' does not exist."
        
        data = torch.load(self.model_path)
        assert 'policy' in data and 'v' in data and 'q' in data, "Loaded data missing 'policy' or 'v' or 'q' keys."
        
        self.policy = data['policy'].numpy()
        self.v = data['v'].numpy()
        self.q = data['q'].numpy()
        print(f"Model loaded from {self.model_path}")
        
    # save the policy, action values and state values to a file    
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        policy_tensor = torch.tensor(self.policy, dtype=torch.float32)
        v_tensor = torch.tensor(self.v, dtype=torch.float32)
        q_tensor = torch.tensor(self.q, dtype=torch.float32)
        torch.save({'policy': policy_tensor, 'v':v_tensor, 'q':q_tensor}, path)
        print(f"Model saved to {path}")