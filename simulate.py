""" 
simulate - functions to simulate a markov chain with given distibutions

For a k-step markov chain, a distribution is given by a k+1 dimensional tensor with
 dimension 0 to k -1 representing the steps t to t-k
 and the kth dimension holding the inverse cdf lookup table.
If the state space contains n disting elements,
 then the shape of the distribution tensor is (n, n, ...,n, m) with
 m being the granularity of the inverse cdf look up table
"""

# %%
import numpy as np

#config
state_type = np.uint8

def path(dist, initial_states, random_numbers):
  '''Returns an array of simulated states (s_0,..s_T) for a distribution
  provided initial states (s_-k,..s_-1) for a k-order markov chain'''
  T = len(random_numbers)
  states = np.zeros(T, dtype=state_type)
  last_states = initial_states
  for i in range(T):
    state = dist[last_states + (random_numbers[i],)]
    states[i] = state
    last_states = last_states[1:] + (state,)
  return states

def random_dist(markov_order, space_size, distribution_granularity):
  dist_dim = markov_order * (space_size,) + (distribution_granularity,)
  dist = np.random.randint(0, space_size, dist_dim, dtype=state_type)
  return dist

def rand_numbers(granularity, length):
  rand_nbrs = np.random.randint(0, granularity, length, dtype=state_type)
  return rand_nbrs

#%% test: deterministic jump

def test_deterministic_jump_process():
  dist = np.array([[1],[0]])
  rns= rand_numbers(1, 10)
  return path(dist, (0,), rns)

def test_2nd_order():
  dist = random_dist(2, 3, 3)
  rns = rand_numbers(1, 10)
  return path(dist, (0,1), rns)

def test_3rd_order():
  space = 2
  dist = random_dist(3, space, space)
  rns = rand_numbers(space, 10)
  return path(dist, (0,0,1), rns)

# print(test_deterministic_jump_process(), test_2nd_order(), test_3rd_order())
