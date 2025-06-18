from config.arguments import args      
import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)
import torch
import numpy as np


class BaseModel:
    def __init__(self, 
                 env,
                 thousands=1e-5,
                 gamma=0.9,
                 model_path = None,
                 env_size=args.env_size, 
                 start_state=args.start_state, 
                 target_state=args.target_state, 
                 forbidden_states=args.forbidden_states):
        self.env = env
        self.thousands = thousands
        self.gamma = gamma
        
        self.model_path = model_path
        self.env_size = env_size
        self.num_states = env_size[0] * env_size[1]
        self.start_state = start_state
        self.target_state = target_state
        self.forbidden_states = forbidden_states

        self.agent_state = start_state
        self.action_space = args.action_space          
        self.reward_target = args.reward_target
        self.reward_forbidden = args.reward_forbidden
        self.reward_step = args.reward_step

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
        
    # load the policy, action values and state values from a file
    def load(self, path):
        self.model_path = path
        assert os.path.exists(self.model_path), f"Model file '{self.model_path}' does not exist."
        
        data = torch.load(self.model_path, weights_only=True)
        assert 'policy' in data and 'v' in data and 'q' in data, "Loaded data missing 'policy' or 'q' keys."
        
        self.policy = data['policy'].numpy()
        self.v = data['v'].numpy()
        self.q = data['q'].numpy()
        print(f"Model loaded from {self.model_path}")
        
    # save the policy, action values and state values to a file    
    def save(self, path, log=False):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        policy_tensor = torch.tensor(self.policy, dtype=torch.float32)
        v_tensor = torch.tensor(self.v, dtype=torch.float32)
        q_tensor = torch.tensor(self.q, dtype=torch.float32)
        torch.save({'policy': policy_tensor, 'v': v_tensor, 'q':q_tensor}, path)
        if log:
            print(f"Model saved to {path}")
