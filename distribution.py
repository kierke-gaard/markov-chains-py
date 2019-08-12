"""
distibtion - basic functionality for distibution handling like
histogram and inverse distribution look ups.
"""

#%% Dependencies and Configuration
import numpy as np
from collections import Counter
from itertools import groupby

state_type = np.uint8

#%% Discrete cdf tables from samples
def histogram(samples):
  '''Returns array of named tuples: 'x' for the realizations, 'count' the number of occurences'''
  cnt = np.array(list(Counter(samples).items()), dtype=[('x', state_type), ('count', float)])
  cnt.sort(order='x')
  m = np.sum(cnt['count'])
  cnt['count'] = cnt['count']/m
  return cnt

def complete_histogram(samples, number_of_states):
  '''Add counts of zeros if there are no observations.
  The index of the returned array represents the observation.'''
  cnt = np.zeros(number_of_states, float)
  hist = histogram(samples)
  cnt[hist['x']] = hist['count']
  return cnt

def inverse_cdf_lookup_table(complete_hist, granularity):
  '''Return a lookup table for state space indices as array.
  The index of the array represent a point in [0,1], namely
  a midpoint of an even partition of #granularity intervals.
  The values in the array represent an index of the statespace.'''
  upper_bounds = np.cumsum(complete_hist)
  mid_points = (np.arange(granularity) + 0.5)/granularity
  arr = np.zeros(granularity, dtype=state_type)
  j, i = 0, 0
  while i < granularity:
    if mid_points[i] > upper_bounds[j]:
      j += 1
    else: #<=
      arr[i] = j
      i +=1
  return arr

def inverse_cdf_lookup(samples, number_of_states, granularity):
  hist_complete = complete_histogram(samples, number_of_states)
  lookup_table_of_inverse_cdf = inverse_cdf_lookup_table(hist_complete, granularity)
  return lookup_table_of_inverse_cdf

def realization(inverse_cdf_lookup_tbl, index):
  return inverse_cdf_lookup_tbl[index]

#%% Retrieve samples from time series
# Assumptions: 1d time series as 1d array with values in state space, no missing values, equitemporal

def sliding_window(arr, window_len=2):
  return np.vstack([arr[i:len(arr) - window_len + i + 1]
                    for i in range(window_len)]).transpose()

def groupby_butlast(arr):
  '''Groups an array with keys as its entries apart from the last dimension
  and value as list of occurences in the last dimension.'''
  butlast = lambda x: x[:-1]
  lasts = lambda xs: [x[-1] for x in xs]
  sorted_arr = sorted(arr, key=butlast)
  butlast_lasts = [(k, lasts(v)) for k, v in groupby(sorted_arr, butlast)]
  return butlast_lasts
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