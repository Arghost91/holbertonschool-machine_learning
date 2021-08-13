#!/usr/bin/env python3
"""
Function that performs the backward algorithm for a hidden markov model
"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    * Observation is a numpy.ndarray of shape (T,) that contains the index
    of the observation
        * T is the number of observations
    * Emission is a numpy.ndarray of shape (N, M) containing the emission
    probability of a specific observation given a hidden state
        * Emission[i, j] is the probability of observing j given the hidden
        state i
        * N is the number of hidden states
        * M is the number of all possible observations
    * Transition is a 2D numpy.ndarray of shape (N, N) containing the
    transition probabilities
        * Transition[i, j] is the probability of transitioning from the
        hidden state i to j
    * Initial a numpy.ndarray of shape (N, 1) containing the probability
    of starting in a particular hidden state
    * Returns: P, B, or None, None on failure
        * P is the likelihood of the observations given the model
        * B is a numpy.ndarray of shape (N, T) containing the backward
        path probabilities
            * B[i, j] is the probability of generating the future observations
            from hidden state i at time j
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
    β = np.zeros((N, T))
    β[:, T - 1] = np.ones((N))
    for i in range(T - 2, -1, -1):
        for j in range(N):
            tran = Transition[j, :]
            em = Emission[:, Observation[i + 1]]
            β[j, i] = np.sum(β[:, i + 1] * tran * em)
    emi = Emission[:, Observation[0]]
    P = np.sum(Initial[:, 0] * emi * β[:, 0])
    return P, β
