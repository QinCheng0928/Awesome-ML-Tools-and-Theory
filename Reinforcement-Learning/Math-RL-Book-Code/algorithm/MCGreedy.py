import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)
from algorithm.model import BaseModel
import numpy as np
import torch
print("cuda is avaliable", torch.cuda.is_available())


class MCGreedy(BaseModel):
    def __init__(self, 
                 env,
                 gamma=0.9,
                 iterations=100,
                 collect_data_steps=100,
                 epsilon=0.1,
                 model_path='../log/MC_greedy/MC_greedy_model_v0.pth',
                 ):
        super().__init__(env, gamma)
        
        self.iterations = iterations
        self.collect_data_steps = round(collect_data_steps)
        self.epsilon = epsilon
        self.model_path = model_path
        self.v = np.zeros(self.num_states)                                               # state values
        self.q = np.zeros((self.num_states, len(self.action_space)))                               # action values
        self.policy = np.ones((self.num_states, len(self.action_space))) / len(self.action_space)  # policy
        self.num = np.zeros((self.num_states, len(self.action_space)))                             # visit counts
        self.returns = np.zeros((self.num_states, len(self.action_space)))                         # returns
    
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
            print(f"Iteration {iterations + 1}: MC Exploring Starts Training...")

            episode = []
            current_state = self.start_state
            for _ in range(self.collect_data_steps):       
                state_idx = self.state_to_index(current_state)
                action_prob = self.policy[state_idx]
                selected_action_idx = np.random.choice(len(action_prob), p=action_prob)
                selected_action = self.action_space[selected_action_idx]

                next_state, reward = self.env.get_next_state_and_reward(current_state, selected_action)
                episode.append((current_state, selected_action, reward))
                current_state = next_state
            
                
            G = 0
            self.returns = np.zeros((self.num_states, len(self.action_space)))
            self.num = np.zeros((self.num_states, len(self.action_space)))   
            for t in range(len(episode) - 1, -1, -1):
                state_t, action_t, reward_t = episode[t]
                G = reward_t + self.gamma * G
                state_index_t = self.state_to_index(state_t)
                action_index_t = self.action_space.index(action_t)

                self.returns[state_index_t, action_index_t] += G
                self.num[state_index_t, action_index_t] += 1
                self.q[state_index_t, action_index_t] = self.returns[state_index_t, action_index_t] / self.num[state_index_t, action_index_t]

                # Update policy with epsilon-greedy
                best_action_index = np.argmax(self.q[state_index_t])
                num_actions = len(self.action_space)
                self.policy[state_index_t, :] = self.epsilon / num_actions
                self.policy[state_index_t, best_action_index] = 1 - self.epsilon + self.epsilon / num_actions
        
        self.v = np.sum(self.policy * self.q, axis=1)       
        print("MC Greedy completed.")