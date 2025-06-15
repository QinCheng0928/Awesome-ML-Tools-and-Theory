import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)
from algorithm.model import BaseModel
import numpy as np
import torch
import random 
print("cuda is avaliable", torch.cuda.is_available())


class MCBasic(BaseModel):
    def __init__(self, 
                 env,
                 gamma=0.9,
                 iterations=100,
                 collect_data_steps=100,
                 model_path='../log/MCBasic/MCBasic_model_v0.pth',
                 ):
        super().__init__(env, gamma)
        
        self.iterations = iterations
        self.collect_data_steps = round(collect_data_steps)
        self.model_path = model_path
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
        for iterations in range(self.iterations):
            print(f"Iteration {iterations + 1}: MC Basic Training...")

            for state_idx in range(self.num_states):
                original_state = self.index_to_state(state_idx)

                for action_index, action in enumerate(self.action_space):
                    current_state = original_state

                    for step in range(self.collect_data_steps):
                        # use current policy to select action
                        selected_action_idx = (action_index if step == 0 else self.policy[self.state_to_index(current_state), :].argmax())
                        selected_action = self.action_space[selected_action_idx]
                        # get next state and reward
                        next_state, reward = self.env.get_next_state_and_reward(current_state, selected_action)
                        # estimate the q value
                        self.q[state_idx, action_index] += reward * (self.gamma ** step)
                        # for next step
                        current_state = next_state  

                # policy improvement
                best_action_index = np.argmax(self.q[state_idx])
                self.policy[state_idx, :] = 0
                self.policy[state_idx, best_action_index] = 1.0

        self.v = np.max(self.q, axis=1)

        print("MC Basic completed.")