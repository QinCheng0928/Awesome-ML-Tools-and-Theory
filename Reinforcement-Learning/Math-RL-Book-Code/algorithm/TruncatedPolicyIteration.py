import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)
from algorithm.model import BaseModel
import numpy as np
import torch
print("cuda is avaliable", torch.cuda.is_available())


class TruncatedPolicyIteration(BaseModel):
    def __init__(self, 
                 env,
                 thousands=1e-5,
                 gamma=0.9,
                 truncated_steps=1000,
                 ):
        super().__init__(env, thousands, gamma)
        
        self.truncated_steps = truncated_steps                             # maximum steps for policy evaluation
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
            t = 0
            while (t < self.truncated_steps):
                last_v = self.v.copy()
                for state_index in range(self.num_states):
                    v_new = 0
                    for action_index, action in enumerate(self.action_space):
                        next_state, reward = self.env.get_next_state_and_reward(self.index_to_state(state_index), action)
                        next_state_index = self.state_to_index(next_state)
                        v_new += self.policy[state_index][action_index] * (reward + self.gamma * last_v[next_state_index])         
                    self.v[state_index] = v_new
                t += 1


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

        print("Truncated Policy Iteration completed.")