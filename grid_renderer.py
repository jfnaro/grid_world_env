import turtle
import numpy as np

class grid_renderer:

    def __init__(self, grid, square_size=50):
        self.grid = grid
        self.square_size = square_size
        self.done_flag = True
        self.height = grid.shape[0]
        self.width = grid.shape[1]
        self.turt = turtle.Turtle()

    def colored_spots(self, blank=0, color_dict={0:'white', 1:'blue', 2:'green', 3:'red', 4:'black', 5:'gray'}):

        self.turt.reset()
        self.turt.hideturtle()
        self.turt.speed(0)
        self.turt.up()

        window_height = self.square_size * self.width
        window_width = self.square_size * self.height
        
        self.x_start = window_width // -2
        x_end = window_width // 2
        self.y_start = window_height // 2
        y_end = window_height // -2

        #draw horizontal lines
        for y_pos in range(y_end, self.y_start + 1, self.square_size):
            self.turt.goto(self.x_start, y_pos)
            self.turt.down()
            self.turt.goto(x_end, y_pos)
            self.turt.up()

        # draw vertical lines
        for x_pos in range(self.x_start, x_end + 1, self.square_size):
            self.turt.goto(x_pos, self.y_start)
            self.turt.down()
            self.turt.goto(x_pos, y_end)
            self.turt.up()

        for y_index in range(self.height):
            y_coord = (y_index * -self.square_size) + self.y_start
            for x_index in range(self.width):
                if self.grid[y_index][x_index] != blank:
                    x_coord = (x_index * self.square_size) + self.x_start
                    self.turt.goto(x_coord, y_coord)
                    self.turt.color(color_dict[self.grid[y_index][x_index]])
                    self.turt.down()
                    self.turt.begin_fill()
                    self.turt.goto(x_coord + self.square_size, y_coord)
                    self.turt.goto(x_coord + self.square_size, y_coord - self.square_size)
                    self.turt.goto(x_coord, y_coord - self.square_size)
                    self.turt.end_fill()
                    self.turt.up()

        if self.done_flag:
            turtle.done()

    def grid_policy(self, policy, color="Deep Pink", up=0, right=1, down=2, left=3, terminal=5):
        self.done_flag = False
        self.colored_spots()
        self.done_flag = True
        self.turt.degrees()

        y_coord = self.y_start - (self.square_size / 2)
        for y_index in range(self.height):
            x_coord = self.x_start + (self.square_size / 2)
            for x_index in range(self.width):
                if policy[y_index][x_index] != terminal:
                    self.turt.goto(x_coord, y_coord)

                    if policy[y_index][x_index] == up:
                        self.turt.setheading(90)
                    elif policy[y_index][x_index] == right:
                        self.turt.setheading(0)
                    elif policy[y_index][x_index] == down:
                        self.turt.setheading(270)
                    elif policy[y_index][x_index] == left:
                        self.turt.setheading(180)

                    self.turt.color(color, color)
                    self.turt.down()
                    self.turt.forward(self.square_size / 2)
                    self.turt.begin_fill()
                    self.turt.right(135)
                    self.turt.forward(self.square_size / 5)
                    self.turt.right(135)
                    self.turt.forward(np.sqrt(2 * ((self.square_size / 5) ** 2)))
                    self.turt.end_fill()
                    self.turt.up()
                
                x_coord += self.square_size
            y_coord -= self.square_size

        turtle.done()

    def grid_returns(self, returns):

        self.done_flag = False
        self.colored_spots()
        self.done_flag = True

        y_coord = self.y_start - (self.square_size / 2)
        for y_index in range(self.height):
            x_coord = self.x_start + (self.square_size / 2)
            for x_index in range(self.width):
                self.turt.goto(x_coord, y_coord)
                self.turt.write(round(returns[y_index][x_index], 4), align='center')
                x_coord += self.square_size
            y_coord -= self.square_size

        turtle.done()
