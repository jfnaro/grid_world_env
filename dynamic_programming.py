#!/usr/bin/env python
# coding: utf-8

# # Dynamic Programming for Reinforcement Learning with Gridworld Examples

# ### This notebook was created to practice applying dynamic programming to reinforcement learning. The notes were mainly intended to help me reflect on what I have learned. If you somehow find this, welcome, and please feel free to contact me at joeynaro@rocketmail.com

# In[1]:


from grid_world_env_deterministic import grid_world_env_deterministic as deterministic_enviro
from grid_renderer import grid_renderer
import numpy as np

np.set_printoptions(suppress=True)#precision=4)


# I have created a few simple gridworld environments. Below I have used one of these environments as the gridworld I am going to solve. The gridworld environments are in my GitHub repo if you are interested in trying them out.

# In[2]:


height = 5
width = 5
hole_id = 4
goal_id = 3
num_of_acts = 4
test = deterministic_enviro()
state, _, _, _ = test.reset()
state = state[0:int(len(state)/2)]
state = np.reshape(state, (height, width))
grid_vis = grid_renderer(state)
grid_vis.colored_spots('blank_environment.eps')


# This is what the environment looks like. The black represent holes, and the red represents the goal
# ![image of environment](./images/blank_environment.png)

# For the first attempt, I have greatly penalized falling in the holes and set the goal to always be zero. Let's see how that goes.

# In[3]:


rewards = np.zeros((height, width))
hole_locations = state == hole_id
goal_location = state == goal_id

hole_penalty = -10
rewards[hole_locations] = hole_penalty
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
            actions[start_row][start_col].append((col, row))


# The starting policy will find each action equally likely, except terminal states, which there is no probability of leaving.

# In[5]:


policy = np.zeros((height, width, num_of_acts))
for row in range(height):
    for col in range(width):
        if (not hole_locations[row][col]) and (not goal_location[row][col]):
            for action in range(num_of_acts):
                policy[row][col][action] = 1.0 / num_of_acts


# For the first exercise, I will not discount future rewards, which should mean that the agent will be looking for future rewards, or just minimized future penalties in this case.

# In[6]:


gamma = 0.9


# The following is an implementation of synchronous value iteration

# In[7]:


values = np.array(rewards)
old_values = np.array(values)

theta = 0.1
delta = 1.0
synchronous_step = 1
while delta > theta:
    for row in range(height):
        for col in range(width):
            next_state_vals = []
            for action in range(num_of_acts):
                next_spot = actions[row][col][action]
                if not (next_spot[0] == col and next_spot[1] == row):
                    next_val = rewards[row][col] + (gamma * old_values[next_spot[1]][next_spot[0]])
                    if next_val != 0:
                        next_state_vals.append(next_val)
            if len(next_state_vals) != 0:
                values[row][col] = max(next_state_vals)
    
    delta = np.amax(abs(np.subtract(old_values, values)))
    print(delta)
    print('step ', synchronous_step)
    synchronous_step += 1
    
    old_values = np.array(values)
    values = np.array(rewards)

    print(old_values)
    
values = old_values

grid_vis.grid_returns(values)


# Looking at the values in the lower left, it looks like agent gets stuck in lower left corner, away from the holes. 

# The code below gives a greedy policy correlating to the values calculated above. For this policy, and all of my gridworld environments, 0 is up, 1 is right, 2 is down, and 3 is left. The 5's below represent terminal states.

# In[ ]:


#TODO: put this in a separate file

new_policy = np.zeros((height, width))

terminal_state = 5

index = 0
for row in range(height):
    for col in range(width):
        if (not hole_locations[row][col]) and (not goal_location[row][col]):
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


# I can check these numbers with linear algebra, though that would be prohibitively resource expensive on a larger scale. The other problem is that the matrix, 1 - the policy matrix, must be invertible. Let's see if that's the case.

# In[ ]:


lin_al_rewards = np.ndarray.flatten(rewards)
lin_al_policy = np.zeros((height * width, height * width))

start_index = 0
for row in range(height):
    for col in range(width):
        if (not hole_locations[row][col]) and (not goal_location[row][col]):
            for next_spot in actions[row][col]:
                next_index = (width * next_spot[1]) + next_spot[0]
                lin_al_policy[start_index][next_index] += (1.0 / num_of_acts)
        start_index += 1

print('Rank: ', np.linalg.matrix_rank(1 - lin_al_policy))
if (lin_al_policy.shape[0] == lin_al_policy.shape[1]) and (np.linalg.matrix_rank(1 - lin_al_policy) == (height * width)):
    lin_al_values = np.matmul(np.linalg.inv(1 - (gamma * lin_al_policy)), lin_al_returns)
    print(lin_al_values)
else:
    print('Matrix is singular')


# While disappointing, it this not unexpected, because of the environment's states. Let's try something new.

# It would appear that in my first attempt, the agent went to the safety of the goal to avoid being near the holes in most cases. However, the agent calculated that the best thing to do in the lower left was to stay there indefinitely. To penalize this, I want to add a penalty for every step taken.

# In[ ]:


step_penalty = -1

for row in range(height):
    for col in range(width):
        if (not hole_locations[row][col]) and (not goal_location[row][col]):
            rewards[row][col] = step_penalty

print(rewards)


# In[ ]:


values = np.array(rewards)
old_values = np.array(values)

theta = 0.1
delta = 1.0
synchronous_step = 1
while delta > theta:
    for row in range(height):
        for col in range(width):
            next_state_vals = []
            for action in range(num_of_acts):
                next_spot = actions[row][col][action]
                if not (next_spot[0] == col and next_spot[1] == row):
                    next_val = rewards[row][col] + (gamma * old_values[next_spot[1]][next_spot[0]])
                    if next_val != 0:
                        next_state_vals.append(next_val)
            if len(next_state_vals) != 0:
                values[row][col] = max(next_state_vals)
    
    delta = np.amax(abs(np.subtract(old_values, values)))
    print(delta)
    print('step ', synchronous_step)
    synchronous_step += 1
    
    old_values = np.array(values)
    values = np.array(rewards)

    print(old_values)
    
values = old_values


# In[ ]:


#TODO: put this in a separate file

new_policy = np.zeros((height, width))

terminal_state = 5

index = 0
for row in range(height):
    for col in range(width):
        if (not hole_locations[row][col]) and (not goal_location[row][col]):
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

grid_vis.grid_policy(new_policy)

# In retrospect, these results are quite humorous. Because the holes have fixed penalties and are terminal states, falling into the holes will cause the agent to take a penalty, but because these states are terminal, the agent will receive no more penalties from taking steps. As such, it appears that, given the current constraints, the agent calculates that its best course of action in many states is to jump into the nearest hole and be done with it.

# In[ ]:


gamma = 0.95

revised_policy = np.zeros((height, width, num_of_acts))

for row in range(height):
    for col in range(width):
        for action in range(num_of_acts):
            revised_policy[row][col][action] = 1.0 / num_of_acts

values = np.array(rewards)
old_values = np.array(values)
print(values)

theta = 0.1
delta = 1.0
synchronous_step = 1
while delta > theta:
    for row in range(height):
        for col in range(width):
            for action in range(num_of_acts):
                next_spot = actions[row][col][action]
                values[row][col] += (revised_policy[row][col][action] * (rewards[row][col] + (gamma * old_values[next_spot[1]][next_spot[0]])))
    
    delta = np.amax(abs(np.subtract(old_values, values)))
    print(delta)
    print('step ', synchronous_step)
    synchronous_step += 1
    
    old_values = np.array(values)
    values = np.array(rewards)

    print(old_values)
    
values = old_values


# In[ ]:


gamma = 0.9

step_penalty = -1

for row in range(height):
    for col in range(width):
        if (not hole_locations[row][col]) and (not goal_location[row][col]):
            rewards[row][col] = step_penalty
            
hole_penalty = -10
rewards[hole_locations] = hole_penalty
print(rewards)

revised_policy = np.zeros((height, width, num_of_acts))

for row in range(height):
    for col in range(width):
        for action in range(num_of_acts):
            revised_policy[row][col][action] = 1.0 / num_of_acts

values = np.array(rewards)
old_values = np.array(values)
print(values)

theta = 0.1
delta = 1.0
synchronous_step = 1
while delta > theta:
    for row in range(height):
        for col in range(width):
            for action in range(num_of_acts):
                next_spot = actions[row][col][action]
                values[row][col] += (revised_policy[row][col][action] * (rewards[row][col] + (gamma * old_values[next_spot[1]][next_spot[0]])))
    
    delta = np.amax(abs(np.subtract(old_values, values)))
    print(delta)
    print('step ', synchronous_step)
    synchronous_step += 1
    
    old_values = np.array(values)
    values = np.array(rewards)

    print(old_values)
    
values = old_values


# In[ ]:




