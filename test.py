import gymnasium
import gym_maze
from gym_maze.utils.play_discrete import play_discrete

env = gymnasium.make('gym_maze/Maze-v0', render_mode="rgb_array", height_range=[5, 20], width_range=[5, 20])
play_discrete(env, noop=-1)