from simulate import *
from distribution import *
import numpy as np

#%% test simulation

def test_deterministic_jump_process():
  dist = np.array([[1],[0]]) # 0->1, 1->0
  initial_states = (0,)
  return simulate(dist, initial_states, 10)

def test_2nd_order():
  dist = random_dist(2, 3, 3)
  return simulate(dist, (0,1), 10)

def test_3rd_order():
  return simulate(random_dist(3, 2, 2), (0,0,1), 10)


#%% test calibration

def sample_usage():
  # Number of different states
  n = 5
  
  # Observations
  samples = np.random.randint(0, n, 10000)
  # build discrete inverse distribution function for lookups
  granularity = n
  inverse_cdf_lookup_tbl = inverse_cdf_lookup(samples, n, granularity)

  # simulate
  random_numbers = np.random.randint(0, n, 10000)
  realizations = inverse_cdf_lookup_tbl.take(random_numbers)
  print('the following two histograms of samples and simulations should be similar')
  print(complete_histogram(samples, n))
  print(complete_histogram(realizations, n))
  
  print('distribution tensor example')
  samples = np.random.randint(0,4,20)
  print(samples, '\n', dist_tensor(samples, 4, 1, 4))


#%% test calibration and simulation

def sample_markov_chain_estimation_simulation():
  
  # calibration data
  n = 5
  samples = np.random.randint(0, n, 3)

  # calibrate 1-order markov process
  order = 1
  granu = 2 * n
  dist = dist_tensor(samples, n, order, granu)

  # simulate
  init_states = (0,)
  sim = simulate(dist, init_states, 100000)
  print('Test calibration and simulation: distribution should be similar')
  print('first order chain')
  print('samples\n', samples[:100], '..')
  print('distribution of in samples\n', dist)
  print('distribution of simulations\n', dist_tensor(sim, n, order, granu))


#%% run tests
print(test_deterministic_jump_process(), test_2nd_order(), test_3rd_order())
sample_usage()
sample_markov_chain_estimation_simulation()