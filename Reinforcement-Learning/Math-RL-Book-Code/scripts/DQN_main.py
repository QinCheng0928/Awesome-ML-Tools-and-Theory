import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)
print("Root directory: ", root_dir)
import time
from grid_world_env.grid_world import GridWorld
from algorithm.DQN import DQN


def train(model_path):
    env = GridWorld()
    model = DQN(
        env,
        gamma=0.99,               # Higher discount factor for DQN
        iterations=1000,          # Typically DQN needs more iterations
        epsilon_start=1.0,        # Start with full exploration
        epsilon_end=0.01,         # Minimum exploration rate
        epsilon_decay=0.995,      # Decay rate for epsilon
        batch_size=64,            # Batch size for experience replay
        buffer_size=10000,        # Size of replay buffer
        target_update_freq=10,    # How often to update target network
        learning_rate=0.001,      # Typically use smaller LR for DQN
        model_path=model_path,
    )
    model.train()
    model.save(model_path)

def evaluate(model_path):
    env = GridWorld()  
    model = DQN(env)
    model.load(model_path)      
    
    for t in range(5):
        state, _ = env.reset()
        done = False
        while not done:
            action = model.predict(state, training=False)
            state, reward, done, info = env.step(action)
            print(f"Step: {t}, Action: {action}, State: {state}, Reward: {reward}, Done: {done}")
            env.render()
        
        # Render the environment
        env.render(animation_interval=2)

def main():
    is_train = False
    model_path = os.path.join(root_dir, 'log/DQN/DQN_model_v0.pth')
    if is_train:
        print("Training the DQN model...")
        train(model_path)
    else:
        print("Evaluating the DQN model...")
        evaluate(model_path)

if __name__ == "__main__":
    main()