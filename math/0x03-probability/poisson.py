#!/usr/bin/env python3
"""
Class that represents a Poisson distribution
"""

class Poisson:
  """
  Class for Poisson
  """
  
    def __init__(self, data=None, lambtha=1.):
        """
        Class Constructor
        """
        self.lambtha = float(lambtha)
        if data is None:
            if lambtha<0:
                raise ValueError("lambtha must be a positive value")
        else:
            self.lambtha = sum(data)/len(data)
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data)<2:
                raise ValueError("data must contain multiple values")              
  
    def pmf(self, k):
        """
        pmf (Probability Mass Function) for a Poisson Distribution
        """
        e = 2.7182818285
        if type(k) is not int:
            k = int(k)
        elif k<0:
            return 0
        facto = 1
        for i in range(1, k+1):
            facto = facto*i
        return ((self.lambtha**k)*(e**-self.lambtha))/facto

    def cdf(self, k):
        """
        cdf (Cumulative Distributive Function) for a Poisson Distribution
        """
        e = 2.7182818285
        sumat = 0
        if type(k) is not int:
            k = int(k)
        elif k<0:
            return 0
        for i in range(k+1):
            facto = 1
            for j in range(1, i+1):
                facto = facto*j
            sumat += ((self.lambtha**i)*(e**-self.lambtha))/facto
        return sumat
