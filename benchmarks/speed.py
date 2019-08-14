import MarkovChain.simulate as sim
import timeit


def benchmark_simulation(params):
  dist = sim.random_dist(params['order'], params['size'], params['granularity'])
  initial_state = params['order'] * (0,)
  s = sim.simulate(dist, initial_state, params["length"])
  print(params)
  return len(s)



params = [
  {"order": 1, "size": 10, "length": 100000, "granularity": 50},
  {"order": 1, "size": 100, "length": 100000, "granularity": 50},
  {"order": 1, "size": 256, "length": 100000, "granularity": 50},
  ]
for p in params:
  print(timeit.timeit('benchmark_simulation(p)', globals=globals(), number=1))
  

# INSIGHTS =======================================================

# => 2 million simulation steps per second and it linear in the number of steps
# {"order": 1, "size": 10, "length": 100000000, "granularity": 20}
# -> 38.7s

# => random number generation takes way less time than simulation
# {'order': 1, 'size': 10, 'length': 10000000, 'granularity': 20}
# 3.7889681000000004
# simulation only: sim.path(dist, initial_state, rnd)
# 3.6840025999999995

# => impact of order is relatively mild
# from 1 to 4 a growth of 20% process time

# => import of granularity almost does not matter

# => impact of size does not matter




