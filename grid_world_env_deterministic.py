import turtle
import numpy as np

#deterministic
class grid_world_env_deterministic:
    """A deterministic gridworld environment"""

    def __init__(self, height=5, width=5, init_pos=(0,0), end_pos=[(-1,-1)], seed=1, holes=[], hole_quantity=5, reward=0, penalty=-10, step_penalty=-1):
        """Create a gridworld environment

        Keyword arguments:
        height -- the height of the grid (default 5)
        width -- the width of the grid (default 5)
        init_pos -- tuple representing initial position (column index, row index) (default (0,0))
        end_pos -- list of goal position tuples (column index, row index) (default (-1,-1))
        seed -- seed for rng used to create holes if none are specified (default 1)
        holes -- list of hole position tuples, will be populated randomly if left empty (default [])
        hole_quantity -- amount of holes to randomly add if holes is empty (default 5)
        reward -- reward for reaching goal (default 0)
        penalty -- penalty for entering hole (default -10)
        step_penalty -- penalty for taking a step (default -1)
        """

        self.empty = 0
        self.agent = 1
        self.goal = 3
        self.hole = 4
        self.reward = reward
        self.penalty = penalty
        self.step_penalty = step_penalty
        self.height = height
        self.width = width
        self.init_pos = init_pos
        self.end_pos = end_pos
        self.x_pos = init_pos[0]
        self.y_pos = init_pos[1]
        self.holes = holes
        self.hole_quantity = hole_quantity
        self.seed_value = seed
        self.square_size = 50
        self.board_turt = turtle.Turtle()

        self.grid = np.zeros((2, self.height, self.width)) # what if these were 1's for a neural network
        self.grid[1][self.init_pos[1]][self.init_pos[0]] = self.agent
        for goal in self.end_pos:
            self.grid[0][goal[1]][goal[0]] = self.goal
        if len(self.holes) == 0 and self.hole_quantity > 0:
            seed = self.seed_value
            np.random.seed(seed)
            for i in range(self.hole_quantity):
                x_rand = np.random.randint(0, self.width)
                y_rand = np.random.randint(0, self.height)
                while (self.grid[0][y_rand][x_rand] != 0) or (x_rand == self.init_pos[0] and y_rand == self.init_pos[1]):
                    x_rand = np.random.randint(0, self.width)
                    y_rand = np.random.randint(0, self.height)
                self.holes.append((x_rand, y_rand))
                self.grid[0][y_rand][x_rand] = self.hole
        else:
            for hole in self.holes:
                self.grid[0][hole[1]][hole[0]] = self.hole

    def reset(self, init_pos=None):
        """Resets environment
        
        Returns
        -the state in a 1 dimensional array
        -a reward of 0
        -a done flag of False
        -an empty documentation dict
        """
        if init_pos != None:
            self.init_pos = init_pos
            self.x_pos = init_pos[0]
            self.y_pos = init_pos[1]
        self.grid[1][self.y_pos][self.x_pos] = self.empty
        self.x_pos = self.init_pos[0]
        self.y_pos = self.init_pos[1]
        self.grid[1][self.y_pos][self.x_pos] = self.agent

        state = np.ndarray.flatten(self.grid)

        return state, 0, False, {}

    def hypothetical(self, state, action):
        """Returns the data from taking an action at a state without actually moving the agent

        Parameters
        state -- coordinate tuple in width-height order
        action -- integer corresponding to action: 0 -> up, 1 -> right, 2 -> down, 3 -> left

        Returns in order
        -the state in a 1 dimensional array
        -a reward of 0
        -a done flag
        -an empty documentation dict
        """

        reward = 0
        done = False
        if 0 <= action <= 3:

            temp_grid = np.array(self.grid)
            x_pos = state[0]
            y_pos = state[1]

            temp_grid[1][y_pos][x_pos] = self.empty
            
            if (action == 0) and (y_pos != 0): #up
                y_pos -= 1
            elif (action == 1) and (x_pos != (self.width - 1)): #right
                x_pos += 1
            elif (action == 2) and (y_pos != (self.height - 1)): #down
                y_pos += 1
            elif (action == 3) and (x_pos != 0): #left
                x_pos -= 1

            temp_grid[1][y_pos][x_pos] = self.agent

            if temp_grid[0][y_pos][x_pos] == self.hole:
                done = True
                reward = self.penalty
            elif temp_grid[0][y_pos][x_pos] == self.goal:
                done = True
                reward = self.reward
            else:
                reward = self.step_penalty

        state = np.ndarray.flatten(temp_grid)

        return state, reward, done, {}

    def step(self, action):
        """Returns the data from taking an action

        Parameters
        action -- integer corresponding to action: 0 -> up, 1 -> right, 2 -> down, 3 -> left

        Returns in order
        -the state in a 1 dimensional array
        -a reward of 0
        -a done flag of False
        -an empty documentation dict
        """

        reward = 0
        done = False
        if 0 <= action <= 3:

            self.grid[1][self.y_pos][self.x_pos] = self.empty
            
            if (action == 0) and (self.y_pos != 0): #up
                self.y_pos -= 1
            elif (action == 1) and (self.x_pos != (self.width - 1)): #right
                self.x_pos += 1
            elif (action == 2) and (self.y_pos != (self.height - 1)): #down
                self.y_pos += 1
            elif (action == 3) and (self.x_pos != 0): #left
                self.x_pos -= 1

            self.grid[1][self.y_pos][self.x_pos] = self.agent

            if self.grid[0][self.y_pos][self.x_pos] == self.hole:
                done = True
                reward = self.penalty
            elif self.grid[0][self.y_pos][self.x_pos] == self.goal:
                done = True
                reward = self.reward
            else:
                reward = self.step_penalty

        state = np.ndarray.flatten(self.grid)

        return state, reward, done, {}

    def render(self, grid=[]): #consider adding the ability to input states
        """Renders the current state of environment and the agent"""

        self.board_turt.clear()
        window_height = self.square_size * self.width
        window_width = self.square_size * self.height
        self.board_turt.hideturtle()
        self.board_turt.speed(0)
        self.board_turt.up()
        
        x_start = window_width // -2
        x_end = window_width // 2
        y_start = window_height // 2
        y_end = window_height // -2

        for y_pos in range(y_end, y_start + 1, self.square_size):
            self.board_turt.goto(x_start, y_pos)
            self.board_turt.down()
            self.board_turt.goto(x_end, y_pos)
            self.board_turt.up()

        for x_pos in range(x_start, x_end + 1, self.square_size):
            self.board_turt.goto(x_pos, y_start)
            self.board_turt.down()
            self.board_turt.goto(x_pos, y_end)
            self.board_turt.up()


        for y_index in range(self.height):
            y_coord = (y_index * -self.square_size) + y_start
            for x_index in range(self.width):
                if self.grid[0][y_index][x_index] != self.empty:
                    x_coord = (x_index * self.square_size) + x_start
                    self.board_turt.goto(x_coord, y_coord)
                    if self.grid[0][y_index][x_index] == self.hole:
                        self.board_turt.color("black")
                    elif self.grid[0][y_index][x_index] == self.goal:
                        self.board_turt.color("red")
                    self.board_turt.down()
                    self.board_turt.begin_fill()
                    self.board_turt.goto(x_coord + self.square_size, y_coord)
                    self.board_turt.goto(x_coord + self.square_size, y_coord - self.square_size)
                    self.board_turt.goto(x_coord, y_coord - self.square_size)
                    self.board_turt.goto(x_coord, y_coord)
                    self.board_turt.end_fill()
                    self.board_turt.up()

        y_coord = (self.y_pos * -self.square_size) + y_start - self.square_size
        x_coord = (self.x_pos * self.square_size) + x_start + (self.square_size / 2)
        self.board_turt.goto(x_coord, y_coord)
        self.board_turt.color("blue")
        self.board_turt.down()
        self.board_turt.begin_fill()
        self.board_turt.circle(self.square_size / 2)
        self.board_turt.end_fill()
        self.board_turt.up()

    def close(self):
        self.board_turt.clear()
