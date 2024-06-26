import numpy as np

class MazeMaker:
    """
        wall_c는 up, down, left, right 순서
    """

    def __init__(self):
        self.dir2idx = {"Up": 0, "Down": 1, "Left": 2, "Right": 3}
        self.dir2move = {"Up": np.array((-1, 0), dtype=int), "Down": np.array((1, 0), dtype=int),
                           "Left": np.array((0, -1), dtype=int), "Right": np.array((0, 1), dtype=int)}
        

    def generate(self, height, width, nr_ratio=0.75):
        wall_c = np.ones((height, width, 4), dtype=bool)
        cell_visited = np.zeros((height, width), dtype=bool)
        cell_stack = []

        # [step 1] Select initial cell and stack.
        cell_current = np.array((np.random.randint(height), np.random.randint(width)), dtype=int)
        cell_stack.append(cell_current)
        cell_visited[cell_current[0], cell_current[1]] = True
        
        # [step 2] Remove wall
        while cell_stack:
            # [step 2.1] Select cell_current; Newest vs. Random
            if np.random.rand() <= nr_ratio:
                cell_current_idx = -1       # Newest; Recursive Backtracking
            else:
                cell_current_idx = np.random.randint(len(cell_stack))       # Random; Prim's Algorithm
            cell_current = cell_stack[cell_current_idx]

            # [step 2.2] Check adjacent cells & remove wall
            for direction in np.random.permutation(["Up", "Down", "Left", "Right"]):
                cell_next = cell_current + self.dir2move[direction]
                # [2.2.a] 방문하지 않은 cell이면, stack cell_next, visit check, remove wall
                if cell_next[0] >= 0 and cell_next[0] < height and cell_next[1] >= 0 and cell_next[1] < width and not cell_visited[cell_next[0], cell_next[1]]:
                    cell_stack.append(cell_next)
                    cell_visited[cell_next[0], cell_next[1]] = True
                    wall_c[cell_current[0], cell_current[1], self.dir2idx[direction]] = False
                    idx_opposite = 2*(self.dir2idx[direction]//2) + (self.dir2idx[direction] + 1)%2
                    wall_c[cell_next[0], cell_next[1], idx_opposite] = False
                    break
                # [2.2.b] (no code) 이미 방문된 셀이면, 다음 방향 순서로 반복

            # [2.2.c] 더 이상 방문할 셀이 없으면, stack에서 현재 cell 제거. break 없이 for 반복을 완료한 상태이므로 else로 진입하게 된다.
            else:
                del cell_stack[cell_current_idx]

        return width, height, wall_c


    def cell2wall(self, wall_c):
        height, width, _ = wall_c.shape
        wall_h = np.ones((height + 1, width), dtype=bool)
        wall_v = np.ones((height, width + 1), dtype=bool)
        
        for i, line in enumerate(wall_c):
            for j, walls in enumerate(line):
                if not walls[self.dir2idx["Up"]]:
                    wall_h[i][j] = False
                if not walls[self.dir2idx["Left"]]:
                    wall_v[i][j] = False
                # Edge Conditions
                if i == height - 1 and not walls[self.dir2idx["Down"]]:
                    wall_h[i + 1][j] = False
                if j == width - 1 and not walls[self.dir2idx["Right"]]:
                    wall_v[i][j + 1] = False

        return width, height, wall_h, wall_v
    

    def wall2cell(self, wall_h, wall_v):
        height, width = wall_h.shape
        height -= 1
        wall_c = np.ones((height, width, 4), dtype=bool)

        # Carve Horizontal
        for i, line in enumerate(wall_h[1:-1]):
            for j, wall in enumerate(line):
                if not wall:
                    wall_c[i][j][self.dir2idx["Down"]] = False
                    wall_c[i + 1][j][self.dir2idx["Up"]] = False
                    # 0, -1번째 줄은 반복문에서 없으므로, 항상 (i + 1)th cell은 존재할 것.
        
        # Carve Vertical
        for i, line in enumerate(wall_v):
            for j, wall in enumerate(line[1:-1]):
                if not wall:
                    wall_c[i][j][self.dir2idx["Right"]] = False
                    wall_c[i][j + 1][self.dir2idx["Left"]] = False

        # Carve Edge Walls; 꼭 할 필요는 없지만, 혹시 확장하면서 버그가 생길지 모르니까 추가
        for i, wall in enumerate(wall_h[0]):
            if not wall: 
                wall_c[0][i][self.dir2idx["Up"]] = False
        for i, wall in enumerate(wall_h[-1]):
            if not wall:
                wall_c[-1][i][self.dir2idx["Down"]] = False
        for i, wall in enumerate(wall_v[:, 0]):
            if not wall: 
                wall_c[i][0][self.dir2idx["Left"]] = False
        for i, wall in enumerate(wall_v[:, -1]):
            if not wall: 
                wall_c[i][-1][self.dir2idx["Right"]] = False

        return width, height, wall_c

    
print()
mazer = MazeMaker()
w, h, wall_c = mazer.generate(10, 10)

print(wall_c)