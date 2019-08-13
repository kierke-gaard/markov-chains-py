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

def rand_numbers(number_of_states, length):
  return np.random.randint(0, number_of_states, length, dtype=state_type)

def simulate(dist, initial_states, length):
  granularity = dist.shape[-1]
  return path(dist, initial_states, rand_numbers(granularity, length))