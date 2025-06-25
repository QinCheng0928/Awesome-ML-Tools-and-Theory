import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)
print("Root directory: ", root_dir)
from grid_world_env.grid_world import GridWorld
from algorithm.PPO import PPO  # Changed from REINFORCE to PPO

def train(model_path):
    env = GridWorld()
    model = PPO(
        env,
        gamma=0.9,               
        iterations=1e2,          
        learning_rate=1e-3,      
        model_path=model_path,    
    )
    model.train()
    model.save(model_path)

def evaluate(model_path):
    env = GridWorld()  
    model = PPO(env)
    model.load(model_path)        
    
    for t in range(5):
        state, _ = env.reset()
        done = False
        while not done:
            action = model.predict(state, training=False)
            state, reward, done, info = env.step(action)
            print(f"Step: {t}, Action: {action}, State: {state}, Reward: {reward}, Done: {done}")
            env.render()
        env.render(animation_interval=2)

def main():
    is_train = True
    model_path = os.path.join(root_dir, 'log/PPO/PPO_model_v0.pth')
    
    if is_train:
        print("Training the PPO model...")
        train(model_path)
    else:
        print("Evaluating the PPO model...")
        evaluate(model_path)

if __name__ == "__main__":
    main()