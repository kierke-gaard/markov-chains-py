#%% 
import numpy as np
from collections import Counter

n = 6
size = 6
samples = np.random.randint(0, n, size)

def integer_histogram(samples, n):
  """ integer_histogram([0, 0, 0, 2, 2], 3) -> [3, 0, 2] """
  cnt = Counter(samples)
  missing_x = set(range(n)) - set(cnt.keys())
  zero_counts = list(zip(missing_x, len(missing_x) * [0]))
  counts = cnt.most_common()
  hist = np.concatenate((counts, zero_counts))
  sorted_hist = sorted(hist, key=lambda x: x[0])
  return [x[1] for x in sorted_hist]

hist = integer_histogram([1,3,3,4],n)

# def cdf_table(hist):
acc = np.cumsum(hist)
m = np.max(acc)
normed = acc / m * (n-1)
print(hist, acc, normed)
f = [int(x) for x in np.round(normed)]
x_f = list(zip(f, range(n)))
# take min of seconds, fill forwards <- pandas




