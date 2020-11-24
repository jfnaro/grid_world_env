from grid_world_env_deterministic import grid_world_env_deterministic as enviro #deterministic
#from grid_world_env_stochastic import grid_world_env_stochastic as enviro #hole with a pull
#from grid_world_env_v2 import grid_world_env_v2 #holes and ice
import turtle
import numpy as np

width = 5
board_size = 25

test = enviro()
done = False

test.close()
test.reset()
test.render()

while not done:
    action = int(input('Enter direction: '))
    state, reward, done, _ = test.step(action)
    formatted_state = ''
    for index in range(len(state)):
        formatted_state += str(state[index]) + ' '
        if (index + 1) % width == 0:
            formatted_state += '\n'
        if (index + 1) % board_size == 0:
            formatted_state += '\n'
    print('---------')
    print('state = \n', formatted_state)
    print('reward = ', reward)
    print('done = ', done)
    print('---------')
    test.close()
    test.render()

turtle.done()