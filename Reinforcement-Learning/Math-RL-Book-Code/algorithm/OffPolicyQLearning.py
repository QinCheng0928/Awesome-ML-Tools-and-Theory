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
                 steps=100,
                 epsilon=0.1,
                 alpha=0.1,
                 model_path='../log/OffPolicy_QLearning/QLearning_model_v0.pth',
                 ):
        super().__init__(env, gamma)
        
        self.steps = round(steps)
        self.epsilon = epsilon
        self.alpha = alpha
        self.model_path = model_path
        self.v = np.zeros(self.num_states)                                                         # state values
        self.q = np.zeros((self.num_states, len(self.action_space)))                               # action values
        self.policy = np.ones((self.num_states, len(self.action_space))) / len(self.action_space)  # policy
        self.num = np.zeros((self.num_states, len(self.action_space)))                             # visit counts
        self.returns = np.zeros((self.num_states, len(self.action_space)))                         # returns
    
    def train(self):
        episode = []
        # Only get one episode
        # If needed, you can modify this to collect multiple episodes.
        state_t = self.start_state
        for _ in range(self.steps):
            state_idx = self.state_to_index(state_t)  
            action_prob = self.policy[state_idx]
            action_idx = np.random.choice(len(action_prob), p=action_prob)
            action_t = self.action_space[action_idx]
            state_t1, reward_t1 = self.env.get_next_state_and_reward(state_t, action_t)
             
            # Store the episode
            episode.append((state_t, action_t, reward_t1, state_t1))              
                
            # Update state
            state_t = state_t1
        for state_t, action_t, reward_t1, state_t1 in episode:
            state_idx = self.state_to_index(state_t)
            state_t1_idx = self.state_to_index(state_t1)
            action_idx = self.action_space.index(action_t)
            
            # update action value
            target = reward_t1 + self.gamma * np.max(self.q[state_t1_idx, :])
            self.q[state_idx, action_idx] -= self.alpha * (self.q[state_idx, action_idx] - target)
            
            # update policy
            best_action_idx = np.argmax(self.q[state_idx])
            self.policy[state_idx, :] = 0
            self.policy[state_idx, best_action_idx] = 1
      
        
        self.v = np.sum(self.policy * self.q, axis=1)       
        print("Q-Learning completed.")