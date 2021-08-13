#!/usr/bin/env python3
"""
Function that performs the Baum-Welch algorithm for
a hidden markov model
"""
import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    * Observations is a numpy.ndarray of shape (T,) that contains
    the index of the observation
        * T is the number of observations
    * Transition is a numpy.ndarray of shape (M, M) that contains
    the initialized transition probabilities
        * M is the number of hidden states
    * Emission is a numpy.ndarray of shape (M, N) that contains
    the initialized emission probabilities
        * N is the number of output states
    * Initial is a numpy.ndarray of shape (M, 1) that contains
    the initialized starting probabilities
    * iterations is the number of times expectation-maximization
    should be performed
    * Returns: the converged Transition, Emission, or None, None on failure
    """
    if type(Observation) is not np.ndarray:
        return None, None
    if type(Emission) is not np.ndarray:
        return None, None
    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return None, None
    N = Transition.shape[0]
    if Transition.shape[0] != Transition.shape[1]:
        return None, None
    if type(Initial) is not np.ndarray or Initial.shape[0] != N:
        return None, None
    T = Observation.shape[0]
