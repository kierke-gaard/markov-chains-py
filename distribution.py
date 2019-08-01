import numpy as np
from collections import Counter, OrderedDict
from itertools import groupby

#%% Discrete cdf tables

def fill_range(index_values, n, fill_value):
  '''[(0, 0), (1, 1), (4, 3), (5, 4)] -> {0: 0, 1: 1, 2: 0, 3: 0, 4: 3, 5: 4}'''
  ind_vals = dict(index_values)
  fill_vals = OrderedDict(zip(range(n), np.full(n, fill_value)))
  fill_vals.update(ind_vals)
  return list(fill_vals.items())

def histogram(samples):
  '''[0, 0, 0, 2, 2] -> {0: 3, 2:2}'''
  cnt = Counter(samples)
  return dict(cnt)

def integer_histogram(samples, n):
  '''[0, 0, 0,  2, 2], 3 -> [3, 0, 2]'''
  hist = fill_range(histogram(samples), n, 0)
  hist_vals = [x[1] for x in hist]
  return hist_vals

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

# %%inverse cdf lookup tables

def remove_consequent_duplicates(list_of_tuples, index):
  '''removes consequent tuples if they equal at given index'''
  return np.array([x for i, x in enumerate(list_of_tuples)
                   if i == 0 or x[index] != list_of_tuples[i-1][index]])

def forward_fill(arr):
    '''Returns an 1d array with nans replaced by forward fill. Heading nans remains
     eg. [nan 0 nan 1 nan] -> [nan 0 0 1 1]'''
    if not isinstance(arr, np.ndarray):
      arr = np.array(arr)
    mask = ~np.isnan(arr)
    idx = np.where(mask, np.arange(len(arr)), 0)
    np.maximum.accumulate(idx, out=idx)
    return arr[idx]

def swap(list_of_pairs):
  return [(x[1], x[0]) for x in list_of_pairs]

def vals(list_of_tuples, index):
  return [x[index] for x in list_of_tuples]

def inverse_cdf_lookup_table(cdf_tab):
  cdf_gaps = remove_consequent_duplicates(cdf_tab, 1)
  inverse_cdf_gaps = swap(cdf_gaps)
  inverse_cdf_nan = fill_range(inverse_cdf_gaps, len(cdf_tab), np.nan)
  inverse_cdf_nan_vals = vals(inverse_cdf_nan, 1)
  inverse_cdf = forward_fill(inverse_cdf_nan_vals)
  return inverse_cdf.astype(int)
