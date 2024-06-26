import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

from maze_maker import MazeMaker


class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}     # * NOTE. 나중에 언제 쓰이는지 체크하기

    def __init__(self, render_mode=None, maze_width=10, maze_height=10):
        self.maze_width = maze_width
        self.maze_width = maze_height
        self.window_size = 720          # Pygame window size

        # Create a placeholder? for Observation Space - locations of the agent and the goal
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(
                    np.array([0, 0]), 
                    np.array([self.maze_height - 1, self.maze_width - 1]), 
                    shape=(2,),
                    dtype=int
                    ),
                "goal": spaces.Box(
                    np.array([0, 0]), 
                    np.array([self.maze_height - 1, self.maze_width - 1]), 
                    shape=(2,),
                    dtype=int
                    )
            }
        )

        # Create a placehoder for Action Space - "Up", "Down", "Left", "Right"
        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array((-1, 0), dtype=int),        # Up
            1: np.array((1, 0), dtype=int),         # Down
            2: np.array((0, -1), dtype=int),        # Left
            3: np.array((0, 1), dtype=int),         # Right
            }

        # Rendering settings
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Create placeholders for render_mode="human"
        self.window = None      # a reference(참조) to the window that we draw to
        self.clock = None       # clock for correct framerate

"""
    def _get_obs(self):








def render(wall_c, w=10, h=10, screen_w=720, screen_h=720):
    pygame.init()
    pygame.display.init()
    screen = pygame.display.set_mode((screen_width, screen_height))

    clock = pygame.time.Clock()

    canvas = pygame.Surface((screen_width, screen_height))
    canvas.fill((255, 255, 255))
    pix_square_size = screen_width/size

    pygame.draw.rect(
        canvas,
        (255, 0, 0),
        pygame.Rect(
            (pix_square_size, pix_square_size), 
            (pix_square_size, pix_square_size),
        ),
    )

    pygame.draw.circle(
        canvas,
        (0, 0, 255),
        ((3 + 0.5) * pix_square_size, (3 + 0.5) * pix_square_size), 
        pix_square_size / 3
    )
    screen.blit(canvas, canvas.get_rect())
    pygame.display.update()
    pygame.time.delay(2000)
    pygame.display.quit()
    pygame.quit()


maze_maker = MazeMaker()
w, h, wall_c = maze_maker.generate(10, 10)

"""