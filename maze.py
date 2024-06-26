import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

#from maze_maker import MazeMaker


class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}     # * NOTE. 나중에 언제 쓰이는지 체크하기

    def __init__(self, render_mode=None, height_range=[5, 10], width_range=[5, 10]):
        self.height_range = height_range
        self.width_range = width_range
        self.window_size = 720          # Pygame window size

        # Create a placeholder? for Observation Space - locations of the agent and the target
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(
                    np.array([0, 0]), 
                    np.array([self.height_range[1] - 1, self.width_range[1] - 1]), 
                    shape=(2,),
                    dtype=int
                    ),
                "target": spaces.Box(
                    np.array([0, 0]), 
                    np.array([self.height_range[1] - 1, self.width_range[1] - 1]), 
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


    def _get_obs(self):
        return {"maze": self.maze, "agent": self._agent_location, "target": self._target_location}


    def reset(self, seed=None, nr_ratio=0.75):
        super().reset(seed=seed)

        # initialize maze size       
        self.maze_height = self.np_random.integers(self.height_range[0], self.height_range[1])
        self.maze_width = self.np_random.integers(self.width_range[0], self.width_range[1])

        # make maze
        self.maze = self.generate_maze(self.maze_height, self.maze_width, nr_ratio=nr_ratio)
        
        # initialize object locations
        self._agent_location = self.np_random.integers([self.maze_height, self.maze_width], size=2, dtype=int)
        self._target_location = self._agent_location
        while np.array_equal(self._agent_location, self._target_location):
            self._target_location = self.np_random.integers([self.maze_height, self.maze_width], size=2, dtype=int)

        # RETURNS
        observation = self._get_obs()
        
        #if self.render_mode == "human":
        #    self._render_frame()

        return observation
        

    def generate_maze(self, height, width, nr_ratio=0.75):
        wall_c = np.ones((height, width, 4), dtype=bool)
        cell_visited = np.zeros((height, width), dtype=bool)
        cell_stack = []

        # [step 1] Select initial cell and stack.
        cell_current = self.np_random.integers([self.maze_height, self.maze_width], size=2, dtype=int) # modified: np.random.randint -> np_random.integers
        cell_stack.append(cell_current)
        cell_visited[cell_current[0], cell_current[1]] = True
        
        # [step 2] Remove wall
        while cell_stack:
            # [step 2.1] Select cell_current; Newest vs. Random
            if self.np_random.random() <= nr_ratio:    # modified: np.random.rand -> self.np_random.random
                cell_current_idx = -1       # Newest; Recursive Backtracking
            else:
                cell_current_idx = self.np_random.integers(len(cell_stack))       # Random; Prim's Algorithm, modified: np.random.randint -> self.np_random.integers
            cell_current = cell_stack[cell_current_idx]

            # [step 2.2] Check adjacent cells & remove wall
            #for direction in np.random.permutation(["Up", "Down", "Left", "Right"]):
            for direction_idx in self.np_random.permutation(4):        # modified: np.random.permutation(["Up", "Down", "Left", "Right"]) -> self.np_random.permutation(4)
                direction = self._action_to_direction[direction_idx]   # direction_idx {0, 1, 2, 3} = direction {up, down, left, right}
                cell_next = cell_current + direction    #self.dir2move[direction]
                # [2.2.a] 방문하지 않은 cell이면, stack cell_next, visit check, remove wall
                if cell_next[0] >= 0 and cell_next[0] < height and cell_next[1] >= 0 and cell_next[1] < width and not cell_visited[cell_next[0], cell_next[1]]:
                    cell_stack.append(cell_next)
                    cell_visited[cell_next[0], cell_next[1]] = True
                    wall_c[cell_current[0], cell_current[1], direction_idx] = False
                    direction_idx_opposite = 2*(direction_idx//2) + (direction_idx + 1)%2
                    wall_c[cell_next[0], cell_next[1], direction_idx_opposite] = False
                    break
                # [2.2.b] (no code) 이미 방문된 셀이면, 다음 방향 순서로 반복

            # [2.2.c] 더 이상 방문할 셀이 없으면, stack에서 현재 cell 제거. break 없이 for 반복을 완료한 상태이므로 else로 진입하게 된다.
            else:
                del cell_stack[cell_current_idx]

        return wall_c


    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pass
"""

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

env = MazeEnv(height_range=[2, 5], width_range=[4, 5])

Hs = np.zeros(12)
Ws = np.zeros(12)

for i in range(1):
    env.reset()
    