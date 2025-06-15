import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)
print("Root directory: ", root_dir)
import time
from grid_world_env.grid_world import GridWorld
from algorithm.PolicyIteration import PolicyIteration


def train(model_path):
    env = GridWorld()
    model = PolicyIteration(
        env,
        thousands=1e-5,
        gamma=0.9,
        )
    model.train()
    model.save(model_path)

def evaluate(model_path):
    env = GridWorld()  
    model = PolicyIteration(env)
    model.load(model_path)          
    for t in range(5):
        state, _ = env.reset()
        done = False
        while not done:
            action = model.predict(state)
            state, reward, done, info = env.step(action)
            print(f"Step: {t}, Action: {action}, State: {state}, Reward: {reward}, Done: {done}")
            env.render()
            
        # Add policy and state values to the environment    
        env.add_policy(model.policy)
        env.add_state_values(model.v)
        
        # Render the environment
        env.render(animation_interval=2)

def main():
    is_train = False
    model_path = os.path.join(root_dir, 'log/policy_iteration/policy_iteration_model_v0.pth')
    if is_train:
        print("Training the Policy Iteration model...")
        train(model_path)
    else:
        print("Evaluating the Policy Iteration model...")
        evaluate(model_path)

    

if __name__ == "__main__":
    main()