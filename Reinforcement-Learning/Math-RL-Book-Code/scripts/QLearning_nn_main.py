import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)
print("Root directory: ", root_dir)
import time
from grid_world_env.grid_world import GridWorld
from algorithm.QLearningnn import QLearningNN


def train(model_path):
    env = GridWorld()
    model = QLearningNN(
        env,
        gamma=0.9,
        iterations=100,
        epsilon=0.1,
        alpha=0.1,
        model_path=model_path,
        )
    model.train()
    model.save(model_path)

def evaluate(model_path):
    env = GridWorld()  
    model = QLearningNN(env)
    model.load(model_path)      
    for t in range(5):
        state, _ = env.reset()
        done = False
        while not done:
            action = model.predict(state)
            state, reward, done, info = env.step(action)
            print(f"Step: {t}, Action: {action}, State: {state}, Reward: {reward}, Done: {done}")
            env.render()
        
        # Render the environment
        env.render(animation_interval=2)

def main():
    is_train = True
    model_path = os.path.join(root_dir, 'log/QLearning_nn/QLearning_nn_model_v0.pth')
    if is_train:
        print("Training the Q-Learning model based on value function...")
        train(model_path)
    else:
        print("Evaluating the Q-Learning model based on value function...")
        evaluate(model_path)

if __name__ == "__main__":
    main()