# maze
Gymnasium RL environment for maze game. <br/> <br/>
This game generates a maze, and the player controls the agent to reach the goal point. When the agent reaches the goal, the game ends and a new game begins (A new maze is generated.) <br/> <br/>
| Overview      | Description |
|-------------------|---------------------------------------------------|
| Action Space      | `Discrete(5, start=-1)`|
| Observation Space | "maze": `MultiBinary([H, W, 4])`, <br/> "agent": `Box([0,0], [H_max - 1, W_max - 1], (2,), int)`, <br/>"target": `Box([0,0], [H_max - 1, W_max - 1], (2,), int)` |
| import            | `gymnasium.make("gym_maze/Maze-v0")` |

`H_max`, `W_max`: the maximum values for height and width. These values are determined when the environment is created. (See [Arguments](#arguments))
<br/><br/>



## Installtion
To get started with this project, follow these steps to set up a Conda environment and install the nessessary dependencies.
```bash
# Clone the repository to your local machine.
git clone https://github.com/chointer/maze.git

# Create and Activate a Conda Environment.
conda create --name myenv python=3.11.9
conda activate myenv

# Install Dependencies
cd maze
pip install -r requriements.txt
```

<br/><br/>



## Usage
```python
import gymnasium as gym
import gym_maze

env = gym.make('gym_maze/Maze-v0', render_mode="human", height_range=[2, 5], width_range=[2, 5])

obs, info = env.reset()
for i in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, _, info = env.step(action)
    if terminated:
        env.reset()
env.close()
```

If you want to control the agent manually,
```python
import gymnasium as gym
import gym_maze
from gym_maze.utils.play_discrete import play_discrete

env = gym.make('gym_maze/Maze-v0', render_mode="rgb_array", height_range=[5, 20], width_range=[5, 20])
play_discrete(env, noop=-1)
```
<br/><br/>



## Maze Generation
This environment generates mazes using [Growing Tree Algorithm](https://weblog.jamisbuck.org/2011/1/27/maze-generation-growing-tree-algorithm). <br/>
The parameter of this algorithm, `nr_ratio`, can be controled in the `reset` function which creates mazes. <br/>

The maze size range is set when the environment is created. (See [Arguments](#arguments)) <br/>
`height_range` and `width_range` take integer lists as boundaries for the height and width ranges, respectively. While the size of the maze can change each time it is generated, the size of the `"maze"` observation remains fixed as the largest maze size; (H_max, W_max, 4).
<br/><br/>



## Action Space
| Value | Meaning |
|-------|---------|
| -1	| NOOP	|
| 0	| UP	|
| 1	| DWON	|
| 2	| LEFT	|
| 3	| RIGHT	|

<br/>



## Observation Space
- **"maze"**<br/>
`MultiBinary([H_max, W_max, 4])`<br/>
"maze" contains maze map information. If [h, w, d] element is False, agent cannot move in the d direction at the h-th row w-th column (blocked).

- **"agent"**<br/>
`Box([0,0], [H_max - 1, W_max - 1], (2,), int)`<br/>
"agent" contains the agent's position.

   | Num   | Observation  		| Min 	| Max 	    |
   |-------|----------------------------|-------|-----------|
   | 0 	| Y position of the agent 	| 0	| H_max - 1 |
   | 1 	| X position of the agent 	| 0	| W_max - 1 |

- **"target"**<br/>
`Box([0,0], [H_max - 1, W_max - 1], (2,), int)`<br/>
"target" contains the target's position.

   | Num   | Observation  		| Min 	| Max 	    |
   |-------|----------------------------|-------|-----------|
   | 0 	| Y position of the target 	| 0	| H_max - 1 |
   | 1 	| X position of the target 	| 0	| W_max - 1 |

<br/>



## Rewards
Each time the agent moves, it receives a reward of -0.01.
When the agent reaches the goal, it receives a reward of 1 (the game is terminated).

<br/><br/>



## Starting State
When the `reset` function is run, a maze is randomly generated, the positions of the agent and the goal are randomly set within the maze.
<br/><br/>



## Arguments
```python
import gymnasium as gym
gym.make('gym_maze/Maze-v0', height_range=[5, 10], width_range=[5, 10])
```
- `height_range`: The boundary for the height range of mazes.
- `width_range`: The boundary for the width range of mazes.

It is recommended to set the maximum height and width to `20` due to the rendering setting.<br/>
The minimum height and width must be greater than `1`.
