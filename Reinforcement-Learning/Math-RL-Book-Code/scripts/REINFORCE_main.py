import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)
print("Root directory: ", root_dir)
from grid_world_env.grid_world import GridWorld
from algorithm.REINFORCE import REINFORCE


def train(model_path):
    env = GridWorld()
    model = REINFORCE(
        env,
        gamma=0.99,               
        iterations=100,          
        learning_rate=1e-3,      
        model_path=model_path,    
    )
    model.train()
    model.save(model_path)

def evaluate(model_path):
    env = GridWorld()  
    model = REINFORCE(env)       
    model.load(model_path)        
    
    for t in range(5):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = model.predict(state, training=False)
            next_state, reward = env.get_next_state_and_reward(state, action)
            done = (next_state == env.target_state)
            
            print(f"Step: {t}, Action: {action}, State: {state}, Reward: {reward}, Done: {done}")
            env.render()
            state = next_state
            total_reward += reward
        
        print(f"Episode {t+1}, Total Reward: {total_reward}")
        env.render(animation_interval=2)

def main():
    is_train = True
    model_path = os.path.join(root_dir, 'log/REINFORCEN/REINFORCEN_model_v0.pth')
    
    if is_train:
        print("Training the REINFORCE model...")
        train(model_path)
    else:
        print("Evaluating the REINFORCE model...")
        evaluate(model_path)

if __name__ == "__main__":
    main()