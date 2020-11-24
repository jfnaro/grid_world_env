#!/usr/bin/env python
# coding: utf-8

# # Dynamic Programming for Reinforcement Learning with Gridworld Examples
# 
# 
# ### Topics
# 
# * Policy Iteration
# * Value Iteration
# * Asynchronous Dynamic Programming

# In[1]:


# These are my own
from grid_world_env_deterministic import grid_world_env_deterministic as deterministic_enviro
from grid_renderer import grid_renderer

import turtle
import numpy as np
import random

np.set_printoptions(suppress=True, threshold=10000)


# In[2]:


height = 5
width = 5
hole_id = 4
goal_id = 3
blank_id = 0
agent_id = 1
num_of_acts = 4
terminal_state = 5
evn_state = 0
agent_state = 1
hole_penalty = -10
step_penalty = -1

env = deterministic_enviro()
state, _, _, _ = env.reset()
state = state[0:int(len(state)/2)]
state = np.reshape(state, (height, width))
goal_locations = state == goal_id
hole_locations = state == hole_id
print(state)
renderer = grid_renderer(state)


# This is what the environment looks like. The black represent holes, and the red represents the goal
# ![image of environment](./images/blank_environment.png)

# I can probe the environment to obtain the following matrix, which gives the rewards for choosing an action from each state

# In[3]:


rewards = np.zeros((height, width, num_of_acts))

for y in range(height):
    for x in range(width):
        for action in range(num_of_acts):
            _, rewards[y][x][action], _, _ = env.hypothetical((x,y), action)

print(rewards)


# The following defines the state that will follow each state given an action. This environment is deterministic.

# In[4]:


actions = []
for start_row in range(height):
    actions.append([])
    for start_col in range(width):
        actions[start_row].append([])
        for action in range(num_of_acts):
            row = start_row
            col = start_col
            if action == 0 and row > 0:
                row -= 1
            elif action == 1 and col < width - 1:
                col += 1
            elif action == 2 and row < height - 1:
                row += 1
            elif action == 3 and col > 0:
                col -= 1
            actions[start_row][start_col].append((row, col))


# These are the parameters I will use to start with. Gamma is the discount factor, and theta is how close to no change in values the algorithm will go before it considers policy evaluation complete

# In[5]:


gamma = 0.9
theta = 0.1


# ## Policy Iteration

# ![Policy iteration algorithm](./images/policy_iteration_formula_page_80.png)

# In[6]:


#uncomment to run turtle visualizer
#renderer.grid_policy(policy)


# Here is a visual representation of the optimal policy determined through policy iteration
# ![optimal policy visualized](./images/optimal_policy_visual.png)

# # Gamma

# In[7]:


# gamma_height = 3
# gamma_width = 3
# gamma_env = deterministic_enviro(height=gamma_height, width=gamma_width, hole_quantity=0)
# gamma_state, _, _, _ = env.reset()
# gamma_state = gamma_state[0:int(len(gamma_state)/2)]
# gamma_state = np.reshape(gamma_state, (gamma_height, gamma_width))
# goal_locations = gamma_state == goal_id
# hole_locations = gamma_state == hole_id
# print(gamma_state)
# renderer = grid_renderer(state)


# In[8]:


gamma = 1


# This evaluates the policy 66 times. Let's try something more efficient.

# ![visual representation of policy iteration](./images/policy_iteration_visual.png)

# ## Value Iteration

# ![visual representation of value iteration](./images/value_iteration_visual.png)

# ![value iteration algorithm](./images/value_iteration_formula_page_83.png)

# The following is an implementation of synchronous value iteration

# In[9]:


delta = 1.0

#Values are intialized arbitrarily 
values = np.zeros((height, width))
values[hole_locations] = hole_penalty
old_values = np.array(values)

synchronous_step = 1
while delta > theta:
    for row in range(height):
        for col in range(width):
            if not (goal_locations[row][col] or hole_locations[row][col]):
                next_state_vals = []
                for action in range(num_of_acts):
                    next_spot = actions[row][col][action]
                    if not (next_spot[0] == col and next_spot[1] == row):
                        next_val = rewards[row][col][action] + (gamma * old_values[next_spot[1]][next_spot[0]])
                        next_state_vals.append(next_val)
                if len(next_state_vals) != 0:
                    values[row][col] = max(next_state_vals)
                    
    delta = np.amax(abs(np.subtract(old_values, values)))
    print(delta)
    print('step ', synchronous_step)
    synchronous_step += 1
    
    old_values = np.array(values)
    values = np.zeros((height, width))
    values[hole_locations] = hole_penalty

    print(old_values)
    
values = old_values


# It looks like success, but let's check the policy. The code below gives a greedy policy correlating to the values calculated above. For this policy, and all of my gridworld environments, 0 is up, 1 is right, 2 is down, and 3 is left. The 5's below represent terminal states. Here is a graphical interpretation of it from my gridworld renderer.

# In[10]:


#TODO: put this in a separate file

new_policy = np.zeros((height, width))

index = 0
for row in range(height):
    for col in range(width):
        if not goal_locations[row][col]:
            max_next_state_val = -100000 #this should never naturally occur
            for action in range(num_of_acts):
                if (actions[row][col][action][0] != col or actions[row][col][action][1] != row):
                    next_spot = actions[row][col][action]
                    next_state_val = values[next_spot[1]][next_spot[0]]
                    if next_state_val > max_next_state_val:
                        max_next_state_val = next_state_val
                        new_policy[row][col] = action
        else:
            new_policy[row][col] = terminal_state

        index += 1
        
print(new_policy)


# This matches the optimal policy previously obtained from policy iteration. Policy iteration and value iteration both went through 8 policy improvements. However, under value iteration, there were only 8 policy evaluations under value iteration compared with 66 policy evaluations under policy iteration.

# ## Asychronous Value Iteration

# Thus far, I've been using synchronous value iteration. I want to see if asynchronous value iteration can do the job in fewer iterations.

# In[11]:


delta = 1.0

#Values are intialized arbitrarily 
values = np.zeros((height, width))
values[hole_locations] = hole_penalty
old_values = np.array(values)

asynchronous_step = 1
while delta > theta:
    for row in range(height):
        for col in range(width):
            if not (goal_locations[row][col] or hole_locations[row][col]):
                next_state_vals = []
                for action in range(num_of_acts):
                    next_spot = actions[row][col][action]
                    if not (next_spot[0] == col and next_spot[1] == row):
                        next_val = rewards[row][col][action] + (gamma * old_values[next_spot[1]][next_spot[0]])
                        next_state_vals.append(next_val)
                if len(next_state_vals) != 0:
                    values[row][col] = max(next_state_vals)
    
    delta = np.amax(abs(np.subtract(old_values, values)))
    print(delta)
    print('step ', asynchronous_step)
    asynchronous_step += 1
    
    old_values = np.copy(values)

    print(values)


# The final values for synchronous and asynchronous have come out the same, and it took the same amount of steps. This took me by surprise for a second until I considered the order values are calculated in, left to right and top to bottom, thus visiting the goal last. I'm going to reverse the order and see if there is a difference, which I do expect

# In[12]:


delta = 1.0

#Values are intialized arbitrarily 
values = np.zeros((height, width))
values[hole_locations] = hole_penalty
old_values = np.array(values)

asynchronous_step = 1
while delta > theta:
    for row in range(height-1, -1, -1):
        for col in range(width-1, -1, -1):
            if not (goal_locations[row][col] or hole_locations[row][col]):
                next_state_vals = []
                for action in range(num_of_acts):
                    next_spot = actions[row][col][action]
                    if not (next_spot[0] == col and next_spot[1] == row):
                        next_val = rewards[row][col][action] + (gamma * values[next_spot[1]][next_spot[0]])
                        next_state_vals.append(next_val)
                if len(next_state_vals) != 0:
                    values[row][col] = max(next_state_vals)
    
    delta = np.amax(abs(np.subtract(old_values, values)))
    print(delta)
    print('step ', asynchronous_step)
    asynchronous_step += 1
    
    old_values = np.copy(values)

    print(values)


# There it is. Asynchronous value iteration has achieved the same thing as synchronous value iteration in almost half the amount of steps, 5 compared to 8. However, it required being clever about how to implement the scanning. Out of curiosity, I will now implement asynchronous value iteration with pseudo-random ordering but being sure to visit every space once before moving to the next step. I expect this to be somewhere between the last two exercises.

# In[13]:


delta = 1.0

#Values are intialized arbitrarily 
values = np.zeros((height, width))
values[hole_locations] = hole_penalty
old_values = np.copy(values)

asynchronous_step = 1

states = [(x,y) for x in range(width) for y in range(height) if not goal_locations[y][x]]
random.shuffle(states)

while delta > theta:
    for state in states:
        col, row = state
        if not (goal_locations[row][col] or hole_locations[row][col]):
            next_state_vals = []
            for action in range(num_of_acts):
                next_spot = actions[row][col][action]
                if not (next_spot[0] == col and next_spot[1] == row):
                    next_val = rewards[row][col][action] + (gamma * values[next_spot[1]][next_spot[0]])
                    next_state_vals.append(next_val)
            if len(next_state_vals) != 0:
                values[row][col] = max(next_state_vals)
    
    delta = np.amax(abs(np.subtract(old_values, values)))
    print(delta)
    print('step ', asynchronous_step)
    asynchronous_step += 1
    
    old_values = np.copy(values)
    
    random.shuffle(states)

    print(values)


# After running the above block several times, I can confirm that random state selection takes between 6 and 8 steps. It makes sense that the previous two examples serve as upper and power bounds for efficiency. As such, one would expect randomness to fall in the middle. This might be worth keeping in mind for less intuitive optimal policies in the future.

# ## Final Notes and Things I've Learned
# 
# Interesting things to note:
# 
# I tried making holes terminal states, so the agent's optimal policy was to jump in the nearest hole to end discounting returns.
# 
# When there are no rewards available and no maximum steps, penalize each step taken or add a small penalty to being in each state, except the final state. Because value iteration takes the maximum next value, and the value of the terminal state is 0 by definition, problems will arise. It will attempt to discount 0, which is zero, and state values of 0 will propagate out from the final states with each iteration.
# 
# 

# In[ ]:





# In[ ]:





# In[14]:


from grid_world_env_stochastic import grid_world_env_stochastic as stochastic_enviro
env = stochastic_enviro()
state, _, _, _ = env.reset()
state = state[0:int(len(state)/2)]
state = np.reshape(state, (height, width))
goal_locations = state == goal_id
hole_locations = state == hole_id
print(state)
renderer = grid_renderer(state)
#renderer.colored_spots()


# In[30]:


transition_matrix = np.zeros((num_of_acts, len(state) * len(state[0]), len(state) * len(state[0])))
i = 0
for row in range(len(state)):
    for col in range(len(state[0])):
        for action in range(num_of_acts):
            next_row, next_col = actions[row][col][action]
            if state[next_row][next_col] in (0, 3, 4):
                index = row * len(state[0]) + col 
                transition_matrix[action][i][index] = 1.
            elif state[next_row][next_col] == 5:
                remainder = 1.
                for act in range(num_of_acts):
                    n_row, n_col = actions[next_row][next_col][act]
                    if state[n_row][n_col] == 4:
                        index = n_row * len(state[0]) + n_col 
                        transition_matrix[action][i][index] += 0.25
                        remainder -= 0.25
                index = next_row * len(state[0]) + next_col
                transition_matrix[action][i][index] = remainder
        i += 1


# In[31]:


print(transition_matrix)


# In[37]:


transition_matrix = np.zeros((len(state) * len(state[0]), len(state) * len(state[0])))
i = 0
for row in range(len(state)):
    for col in range(len(state[0])):
        if state[row][col] in (0, 3, 4):
            index = row * len(state[0]) + col 
            transition_matrix[i][index] = 1.
        elif state[row][col] == 5:
            remainder = 1.
            for act in range(num_of_acts):
                n_row, n_col = actions[row][col][act]
                if state[n_row][n_col] == 4:
                    index = n_row * len(state[0]) + n_col 
                    transition_matrix[i][index] += 0.25
                    remainder -= 0.25
            index = row * len(state[0]) + col
            print(row)
            print(len(state[0]))
            print(col)
            print(index)
            transition_matrix[i][index] = remainder
        i += 1


# In[38]:


print(transition_matrix)


# ## Stochasctic Environment

# In[40]:


delta = 1.0
gamma = 0.9

rewards = np.zeros((height, width))
rewards -= 1.
rewards[hole_locations] = -10.
rewards[goal_locations] = 0.


#Values are intialized arbitrarily 
values = np.copy(rewards)
old_values = np.array(values)

asynchronous_step = 1
while delta > theta:
    for row in range(height-1, -1, -1):
        for col in range(width-1, -1, -1):
            if not (goal_locations[row][col] or hole_locations[row][col]):
                next_state_vals = []
                for action in range(num_of_acts):
                    next_row, next_col = actions[row][col][action]
                    if state[next_row][next_col] in (0, 3, 4):
                        next_val = rewards[next_row][next_col] + (gamma * values[next_row][next_col])
                        next_state_vals.append(next_val)
                    else:
                        index = (next_row * len(state[0])) + next_col
                        next_val = 0
                        for s in range(len(state) * len(state[0])):
                            if transition_matrix[index][s] > 0:
                                r = s // len(state[0])
                                c = s % len(state[0])
                                test1 = transition_matrix[index][s]
                                test2 = rewards[r][c]
                                test3 = values[r][c]
                                next_val += transition_matrix[index][s] * (rewards[next_row][next_col] + (gamma * values[r][c]))
                        next_state_vals.append(next_val)
                if len(next_state_vals) != 0:
                    values[row][col] = max(next_state_vals)
    
    delta = np.amax(abs(np.subtract(old_values, values)))
    print(delta)
    print('step ', asynchronous_step)
    asynchronous_step += 1
    
    old_values = np.copy(values)

    print(values)

# In[ ]:




