import turtle
import numpy as np
import time

#the hole has pull
class grid_world_env_stochastic:

    def __init__(   self, 
                    height=5, 
                    width=5, 
                    init_pos=(0,0), 
                    end_pos=(-1,-1), 
                    seed = 1,
                    holes=[], 
                    hole_quantity=5,
                    pull_chance = 0.25):
        self.empty = 0
        self.agent = 1
        self.goal = 3
        self.hole = 4
        self.grav = 5
        self.reward = 1 #reward for winning
        self.penalty = 0 #penalty for losing
        self.height = height
        self.width = width
        self.end_pos = end_pos
        self.init_pos = init_pos
        self.x_pos = init_pos[0]
        self.y_pos = init_pos[1]
        self.pull_chance = pull_chance
        self.holes = holes
        self.hole_quantity = hole_quantity
        self.seed_value = seed
        self.square_size = 50
        self.board_turt = turtle.Turtle()
        np.random.seed(int(time.time()))

    def seed(self, seed_value):
        self.seed_value = seed_value

    def reset(self, init_pos=None):
        if init_pos != None:
            self.init_pos = init_pos
            self.x_pos = init_pos[0]
            self.y_pos = init_pos[1]
        self.grid = np.zeros((2, self.height, self.width)) #[[0] * self.width for _ in range(self.height)] # what if these were 1's for a neural network
        self.grid[1][self.init_pos[1]][self.init_pos[0]] = self.agent
        self.grid[0][self.end_pos[1]][self.end_pos[0]] = self.goal
        if len(self.holes) == 0:
            seed = self.seed_value
            np.random.seed(seed)
            for i in range(self.hole_quantity):
                x_rand = np.random.randint(0, self.width)
                y_rand = np.random.randint(0, self.height)
                while self.grid[0][y_rand][x_rand] != 0:
                    x_rand = np.random.randint(0, self.width)
                    y_rand = np.random.randint(0, self.height)
                self.holes.append((x_rand, y_rand))
                self.grid[0][y_rand][x_rand] = self.hole
                for delta_x in range(-1, 2, 2):
                    if (0 <= (x_rand + delta_x) < self.height) and (self.grid[0][y_rand][x_rand + delta_x] == self.empty):
                        self.grid[0][y_rand][x_rand + delta_x] = self.grav
                for delta_y in range(-1, 2, 2):
                    if (0 <= (y_rand + delta_y) < self.height) and (self.grid[0][y_rand + delta_y][x_rand] == self.empty):
                        self.grid[0][y_rand + delta_y][x_rand] = self.grav
            np.random.seed(int(time.time()))
        else:
            for hole in self.holes:
                self.grid[0][hole[1]][hole[0]] = hole
                for delta_x in range(-1, 2, 2):
                    if (0 <= (hole[0] + delta_x) < self.width) and (self.grid[0][hole[1]][hole[0] + delta_x] == self.empty):
                        self.grid[0][hole[1]][hole[0] + delta_x] = self.grav
                for delta_y in range(-1, 2, 2):
                    if (0 <= (hole[1] + delta_y) < self.height) and (self.grid[0][hole[1] + delta_y][hole[0]] == self.empty):
                        self.grid[0][hole[1] + delta_y][x_rand] = self.grav

        state = np.ndarray.flatten(self.grid)

        return state, 0, False, {}

    def step(self, action):
        # 0 -> up
        # 1 -> right
        # 2 -> down
        # 3 -> left

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
            elif self.grid[0][self.y_pos][self.x_pos] == self.grav:
                test = np.random.random()
                if test < self.pull_chance:
                    state = []
                    info = {}
                    pull_direction = -1
                    while not done:
                        pull_direction += 1
                        state, reward, done, info = self.step(pull_direction)
                    return state, reward, done, info

        state = np.ndarray.flatten(self.grid)

        return state, reward, done, {}

    def render(self):
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
                    elif self.grid[0][y_index][x_index] == self.grav:
                        self.board_turt.color("gray")
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