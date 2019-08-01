#%% 
import numpy as np
from collections import Counter, OrderedDict
from itertools import groupby

def fill_range(index_values, n, fill_value):
  '''[(0, 0), (1, 1), (4, 3), (5, 4)] -> {0: 0, 1: 1, 2: 0, 3: 0, 4: 3, 5: 4}'''
  ind_vals = dict(index_values)
  fill_vals = OrderedDict(zip(range(n), np.full(n, fill_value)))
  fill_vals.update(ind_vals)
  return fill_vals

def histogram(samples):
  '''[0, 0, 0, 2, 2] -> {0: 3, 2:2}'''
  cnt = Counter(samples)
  return dict(cnt)

def integer_histogram(samples, n):
  '''[0, 0, 0,  2, 2], 3 -> [3, 0, 2]'''
  hist = fill_range(histogram(samples), n, 0)
  return list(hist.values())

integer_histogram([0, 0, 0,  2, 2], 4)

def cdf_table(hist, n):
  '''Returns the values of the distribution function as array of pairs.
  Observations in the first entry and cdf in the second.
  The probabilities of cdf are integers from 0 to n-1.'''
  acc = np.cumsum(hist)
  m = np.max(acc)
  normed = acc / m * n
  cdf = np.maximum(np.round(normed).astype(int) - 1, 0)
  x_f = np.stack((range(n), cdf), 1)
  return x_f 

# minor tests
n = 6
size = 10000
samples = np.random.randint(1, n - 1, size)
# hist = integer_histogram([1,3,3,4],n)
hist = integer_histogram(samples, n)
# print(samples)
print(hist)
print(cdf_table(hist, n))

#%%
def remove_consequent_duplicates_by_second_val(list_of_pairs):
  '''[(0, 0), (1, 1), (2, 1), (3, 4), (4, 5), (5, 5)] -> [(0, 0), (1, 1), (3, 4), (4, 5)]'''
  return [x for i, x in enumerate(list_of_pairs) if i == 0 or x[1] != list_of_pairs[i-1][1]]

f_x_removed_cons_dupl = remove_consequent_duplicates(x_f)
remove_consequent_duplicates(x_f)


def forward_fill(arr):
    '''Returns an 1d array with nans replaced by forward fill. Heading nans remains
     eg. [nan 0 nan 1 nan] -> [nan 0 0 1 1]'''
    mask = ~np.isnan(arr)
    idx = np.where(mask, np.arange(len(arr)), 0)
    np.maximum.accumulate(idx, out=idx)
    return arr[idx]

np.round(np.array([1.4, 1.6])).astype(int)
np.stack(([1,2,3], [10, 20, 30]), 1)