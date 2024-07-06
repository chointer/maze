from gymnasium.envs.registration import register

register(
    id="gym_maze/Maze-v0",
    entry_point="gym_maze.envs.maze:MazeEnv",
)