import MarkovChain.simulate as sim
import sys

# @profile
def benchmark_simulation(params):
  dist = sim.random_dist(params['order'], params['size'], params['granularity'])
  initial_state = params['order'] * (0,)
  s = sim.simulate(dist, initial_state, params["length"])
  print('shape dist', dist.shape, 'size kb', sys.getsizeof(dist)/1000)
  print('shape s', s.shape, 'size kb ', sys.getsizeof(s)/1000)
  return len(s)

params = {"order": 3, "size": 100, "length": 10000000, "granularity": 200}

benchmark_simulation(params)

# NOTE: python -m memory_profiler memory.py for detailed mem profiling

# INSIGHTS
# memory is not an issue
# shape dist (100, 100, 100, 200) size kb 200000.144
# 2e8 Datenpunkte * sizeof(unit8) = 200Mb
# shape s (10000000,) size kb  10000.096