__credits__ = ["Intelligent Unmanned Systems Laboratory at Westlake University."]
'''
Specify parameters of the env
'''
from typing import Union
import numpy as np
import argparse

parser = argparse.ArgumentParser("Grid World Environment")

## ==================== User settings ====================
# specify the number of columns and rows of the grid world
# The first element represents the column index (horizontal coordinate), and the second element represents the row index (vertical coordinate).
parser.add_argument("--env-size", type=Union[list, tuple, np.ndarray], default=(5,5) )   

# specify the start state
parser.add_argument("--start-state", type=Union[list, tuple, np.ndarray], default=(2,2))

# specify the target state
parser.add_argument("--target-state", type=Union[list, tuple, np.ndarray], default=(4,4))

# sepcify the forbidden states
parser.add_argument("--forbidden-states", type=list, default=[ (2, 1), (3, 3), (1, 3)] )

# sepcify the reward when reaching target
parser.add_argument("--reward-target", type=float, default = 10)

# sepcify the reward when entering into forbidden area
parser.add_argument("--reward-forbidden", type=float, default = -5)

# sepcify the reward for each step
parser.add_argument("--reward-step", type=float, default = -1)
## ==================== End of User settings ====================


## ==================== Advanced Settings ====================
parser.add_argument("--action-space", type=list, default=[(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)] )  # down, right, up, left, stay           
parser.add_argument("--debug", type=bool, default=False)
parser.add_argument("--animation-interval", type=float, default = 0.2)
## ==================== End of Advanced settings ====================


args = parser.parse_args()     
def validate_environment_parameters(env_size, start_state, target_state, forbidden_states):
    if not (isinstance(env_size, (list, tuple, np.ndarray))) and len(env_size) != 2:
        raise ValueError("Invalid environment size. Expected a tuple (rows, cols) with positive dimensions.")
    for dim in env_size:
        if not (isinstance(dim, int) and dim > 0):
            raise ValueError("Environment dimensions must be positive integers.")
    
    for state_name, state in [("start_state", start_state), ("target_state", target_state)]:
        if not (isinstance(state, (list, tuple, np.ndarray))) or len(state) != 2:
            raise ValueError(f"{state_name} must be a tuple/list/ndarray of length 2.")
        for i in range(2):
            if (0 <= state[i]) and (state[i] < env_size[i]):
                raise ValueError(f"{state_name} coordinate {state[i]} out of bounds for env_size {env_size}.")
            
    if not isinstance(forbidden_states, (list, tuple, np.ndarray)):
        raise ValueError("forbidden_states must be a list/tuple/ndarray of coordinates.")
    
    for idx, fs in enumerate(forbidden_states):
        if not (isinstance(fs, (list, tuple, np.ndarray)) and len(fs) == 2):
            raise ValueError(f"forbidden_states[{idx}] is not a valid coordinate tuple/list.")
        for i in range(2):
            if not (0 <= fs[i]) and (fs[i] < env_size[i]):
                raise ValueError(f"forbidden_states[{idx}] coordinate {fs[i]} out of bounds for env_size {env_size}.")
try:
    validate_environment_parameters(args.env_size, args.start_state, args.target_state, args.forbidden_states)
except ValueError as e:
    print("Error:", e)