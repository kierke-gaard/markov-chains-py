"""
distibtion - basic functionality for distibution handling like
histogram and inverse distribution look ups.
"""
import numpy as np
from collections import Counter

state_type = np.uint8

#%% Discrete cdf tables from samples
def histogram(samples):
  '''Returns array of named tuples: 'x' for the realizations, 'count' the number of occurences'''
  cnt = np.array(list(Counter(samples).items()), dtype=[('x', state_type), ('count', float)])
  cnt.sort(order='x')
  m = np.sum(cnt['count'])
  cnt['count'] = cnt['count']/m
  return cnt

def complete_histogram(hist, number_of_states):
  '''Add counts of zeros if there are no observations'''
  cnt = np.zeros(number_of_states, float)
  cnt[hist['x']] = hist['count']
  return cnt

def inverse_cdf_lookup_table(hist, granularity):
  '''Return a lookup table for state space indices as array.
  The index of the array represent a point in [0,1], namely
  a midpoint of an even partition of #granularity intervals.
  The values in the array represent an index of the statespace.'''
  upper_bounds = np.cumsum(hist['count'])
  mid_points = (np.arange(granularity) + 0.5)/granularity
  arr = np.zeros(granularity, dtype=state_type)
  j, i = 0, 0
  while i < granularity:
    if mid_points[i] > upper_bounds[j]:
      j += 1
    else: #<=
      arr[i] = hist['x'][j]
      i +=1
  return arr

def inverse_cdf_lookup(samples, granularity):
  return inverse_cdf_lookup_table(histogram(samples), granularity)

def realization(inverse_cdf_lookup_tbl, index):
  return inverse_cdf_lookup_tbl[index]

#%% Retrieve samples from time series
# Assumptions: 1d time series as 1d array with values in state space, no missing values, equitemporal

def sliding_window(arr, window_len=1):
  # step_size implicitly set to 1
  n = len(arr)
  return np.vstack([arr[i:n - window_len + i + 1] for i in range(window_len)]).transpose()

#%% sample usage

# Number of different states

def sample_usage():
  # Number of different states
  n_realizations = 10

  # Observations
  samples = np.random.randint(0, n_realizations, 1000)

  # build discrete inverse distribution function for lookups
  granularity = 10
  inverse_cdf_lookup_tbl = inverse_cdf_lookup(samples, granularity)

  # simulate
  random_numbers = np.random.randint(0, n_realizations, 1000)
  realizations = inverse_cdf_lookup_tbl.take(random_numbers)
  return realizations

# print(sample_usage())