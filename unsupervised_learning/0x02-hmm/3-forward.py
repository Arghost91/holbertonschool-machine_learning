#!/usr/bin/env python3
"""
Function that performs the forward algorithm for a hidden markov model
"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    * Observation is a numpy.ndarray of shape (T,) that contains the
    index of the observation
        * T is the number of observations
    * Emission is a numpy.ndarray of shape (N, M) containing the
    emission probability
    of a specific observation given a hidden state
        * Emission[i, j] is the probability of observing j given the
        hidden state i
        * N is the number of hidden states
        * M is the number of all possible observations
    * Transition is a 2D numpy.ndarray of shape (N, N) containing the
    transition probabilities
        * Transition[i, j] is the probability of transitioning from the
        hidden state i to j
    * Initial a numpy.ndarray of shape (N, 1) containing the probability
    of starting in a
    particular hidden state
    * Returns: P, F, or None, None on failure
    * P is the likelihood of the observations given the model
    * F is a numpy.ndarray of shape (N, T) containing the forward path
    probabilities
    * F[i, j] is the probability of being in hidden state i at time j
    given the previous
    observations
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
    α = np.zeros((N, T))
    em = Emission[:, Observation[0]]
    α[:, 0] = Initial.T * em
    for i in range(1, T):
        for j in range(N):
            auxil = α[:, i - 1] * Transition[:, j]
            col = Observation[i]
            α[j, i] = np.sum(auxil * Emission[j, col])
    P = np.sum(α[:, -1])
    return P, α
