# Math-RL-Book-Code

This repository contains code examples and implementations related to the book *Mathematical Foundations of Reinforcement Learning*. It is organized to facilitate learning and experimentation with grid world environments and reinforcement learning algorithms.

---

## Project Structure
```
Math-RL-Book-Code/
│
├── algorithm/
│   ├── model.py                         # The base class for models
│   ├── ValueIteration.py                # Value Iteration Algorithm Model
│   └── .../                             # More models will be added, with similar structures
│
├── config/
│   └── arguments.py                     # Configuration parameter script
│
├── grid_world_env/
│   └── grid_world.py                    # Grid World environment implementation
│
├── log/
│   ├── value_iteration/                 # Save the trained model
│   │   └── value_iteration_model_v0.pth # Corresponding model data
│   └── .../                         
│
├── scripts/
│   ├── value_iteration_main.py          # Implementing the value iteration algorithm
│   └── .../                             # More algorithms will be added, with similar structures
│
├── test/
│   └── example_grid_world.py            # Example code demonstrating simulation
│
├── requirements.txt                     # Python dependency list
└── README.md                            # Project documentation
```
---

## Running the Example

To run the example simulation:

1. Open the repository folder `Math-RL-Book-Code` in VSCode.

2. Navigate to `test/example_grid_world.py`.

3. Run the script:
   ```bash
   python test/example_grid_world.py
