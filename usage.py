import gymnasium as gym
import gym_maze

# 1. Usage
env = gym.make('gym_maze/Maze-v0', render_mode="human", height_range=[2, 5], width_range=[2, 5])

obs, info = env.reset()
for i in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, _, info = env.step(action)
    if terminated:
        env.reset()
env.close()


"""
# 2. Manual controll

from gym_maze.utils.play_discrete import play_discrete

env = gym.make('gym_maze/Maze-v0', render_mode="rgb_array", height_range=[5, 20], width_range=[5, 20])
play_discrete(env, noop=-1)

"""