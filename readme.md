## MarkovChain: Efficient Simulation and Parameter Estimation for Markov Chains

MarkovChain is an efficient python implementation for parameter estimation and simulation of [Markov chains](https://en.wikipedia.org/wiki/Markov_chain). 

It provides:
 - functionality to estimate the transition probabilities
 - an efficient model to store the transition probabilities
 - and an efficient simulation model

Dependencies:
 - Numpy

## Restrictions on the Markov Chain
------------------------------

A Markov Chain is a stochastic process with finite state space and discrete time steps. The distribution of the next state depends only on the present and previous k steps (k is the order of the chains), in other words the distribution of X(t+1) depends on X(t),..X(t-k) and not on even earlier states.

This package makes the following restrictions on Markov Chains:
 - the transition distribution X(t+1)|X(t),..X(t-k) is time homogeneous, i.e. is independent of t
 - the state space are integers from 0 till n-1, if there are n distinct states. Note that this is not really a constraint as every finite set might be mapped to those integers (also multidimensional state spaces).
 - The size of the state space is restricted by the parameter state_type, which is uint8 by default corresponding to 256 distinct states.

 ## Storage model for transition probabilities
 The transition probabilities X(t+1)|X(t),..X(t-k) are not directly stored, but instead an urn of m states from which the next states is drawn. The higher granularity paramter m, the closer the distribution of the urn to the transition probabilities. Each of the n^k possible tuples X(t),..X(t-k) are assigned to an urn represented as a vector of length m. All together is stored as numpy array with the shape (n, n,... ,m) where n appears k times, called urn-tensor.
 This urn-tensor allow for efficient simulation in the following manner
  1. Generate r random numbers out of 0,..n-1 (Eg. efficiently by simulate.rand_numbers(n,r))
  2. Provide initial states X(-k),..X(-1)
  3. Take the next random number i from 1.
  4. Set the next state to X(t+1) = urn-tensor(X(t),..X(t-k),i)
  5. t <- t+1 and go to step 3.

