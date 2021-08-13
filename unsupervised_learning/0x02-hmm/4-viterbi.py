#!/usr/bin/env python3
"""
Function that calculates the most likely sequence of hidden
states for a hidden markov model
"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    * Observation is a numpy.ndarray of shape (T,) that contains
    the index of the observation
        * T is the number of observations
    * Emission is a numpy.ndarray of shape (N, M) containing the
    emission probability of a specific observation given a hidden state
        * Emission[i, j] is the probability of observing j given the
        hidden state i
        * N is the number of hidden states
        * M is the number of all possible observations
    * Transition is a 2D numpy.ndarray of shape (N, N) containing the
    transition probabilities
        * Transition[i, j] is the probability of transitioning from the
        hidden state i to j
    * Initial a numpy.ndarray of shape (N, 1) containing the probability
    of starting in a particular hidden state
    * Returns: path, P, or None, None on failure
        * path is the a list of length T containing the most likely
        sequence of hidden states
        * P is the probability of obtaining the path sequence
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
    Ω = np.zeros((N, T))
    em = Emission[:, Observation[0]]
    auxil = Initial * em.reshape(-1, 1)
    Ω[:, 0] = auxil.reshape(-1)
    backpo = np.zeros((N, T))
    backpo[:, 0] = 0
    for i in range(1, T):
        for j in range(N):
            prev = Ω[:, i - 1]
            tran = Transition[:, j]
            col = Observation[i]
            emi = Emission[j, col]
            res = prev * tran * emi
            Ω[j, i] = np.amax(res)
            backpo[j, i - 1] = np.argmax(res)
    path = []
    last = np.argmax(Ω[:, T - 1])
    path.append(int(last))
    for i in range(T - 2, -1, -1):
        path.append(int(backpo[int(last), i]))
        last = backpo[int(last), i]
    path.reverse()
    P = np.amin(np.amax(Ω, axis=0))
    return path, P
