import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)
from algorithm.model import BaseModel
import numpy as np
import torch
print("cuda is avaliable", torch.cuda.is_available())


class QLearning(BaseModel):
    def __init__(self, 
                 env,
                 gamma=0.9,
                 iterations=100,
                 epsilon=0.1,
                 alpha=0.1,
                 model_path='../log/OnPolicy_QLearning/QLearning_model_v0.pth',
                 ):
        super().__init__(env, gamma)
        
        self.iterations = iterations
        self.epsilon = epsilon
        self.alpha = alpha
        self.model_path = model_path
        self.v = np.zeros(self.num_states)                                                         # state values
        self.q = np.zeros((self.num_states, len(self.action_space)))                               # action values
        self.policy = np.ones((self.num_states, len(self.action_space))) / len(self.action_space)  # policy
        self.num = np.zeros((self.num_states, len(self.action_space)))                             # visit counts
        self.returns = np.zeros((self.num_states, len(self.action_space)))                         # returns
    
    def train(self):
        for iterations in range(self.iterations):
            print(f"Iteration {iterations + 1}: Q-Learning Training...")

            state_t = self.start_state
            while (state_t != self.target_state):
                state_idx = self.state_to_index(state_t)
                action_prob = self.policy[state_idx]
                action_idx = np.random.choice(len(action_prob), p=action_prob)
                action_t = self.action_space[action_idx]
                state_t1, reward_t1 = self.env.get_next_state_and_reward(state_t, action_t)
                state_t1_idx = self.state_to_index(state_t1)
                
                # update action value
                target = reward_t1 + self.gamma * np.max(self.q[state_t1_idx, :])
                self.q[state_idx, action_idx] -= self.alpha * (self.q[state_idx, action_idx] - target)
                
                # update policy
                best_action_idx = np.argmax(self.q[state_idx])
                num_actions = len(self.action_space)
                self.policy[state_idx, :] = self.epsilon / num_actions
                self.policy[state_idx, best_action_idx] = 1 - self.epsilon + self.epsilon / num_actions
                
                
                # Update state
                state_t = state_t1
        
        self.v = np.sum(self.policy * self.q, axis=1)       
        print("Q-Learning completed.")