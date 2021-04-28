#!/usr/bin/env python3
"""
Class that represents a Exponential distribution
"""


class Exponential:
    """
    Class for Exponential
    """
    def __init__(self, data=None, lambtha=1.):
        """
        Class Constructor
        """
        self.lambtha = float(lambtha)
        if data is None:
            if lambtha < 0:
                raise ValueError("lambtha must be a positive value")
        else:
            self.lambtha = 1 / (sum(data) / len(data))
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")         
    
    def pdf(self, x):
        """
        pdf (Probability Density Function) for a Exponential Distribution
        """
        e = 2.7182818285
        if x < 0:
            return 0
        else:
            return self.lambtha * (e ** -(self.lambtha * x))
        
    def cdf(self, x):
        """
        cdf (Cumulative Distribution Function) for a Exponential Distribution
        """
        e = 2.7182818285
        if x < 0:
            return 0
        else:
            return 1 - (e ** -(self.lambtha * x))
