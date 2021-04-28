#!/usr/bin/env python3
"""
Class that represents a Normal distribution
"""


class Normal:
    """
    Class for Normal
    """
    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Class Constructor
        """
        self.mean = float(mean)
        self.stddev = float(stddev)
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
        else:
            self.mean = sum(data) / len(data)
            var = [(x - self.mean) ** 2 for x in data]
            self.stddev = (sum(var) / len(data)) ** (1 / 2)
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")

    def z_score(self, x):
        """
        Calculates z-score of a given x-value
        """
        return (x - self.mean)/self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score
        """
        return (z*self.stddev) + self.mean

    def pdf(self, x):
        """
        pdf (Probability Density Function) for a Normal Distribution
        """
        e = 2.7182818285
        π = 3.1415926536
        return (1 / (self.stddev * (2 * π) ** (1/2)) * e **
                (- (1 / 2) * ((x - self.mean) / self.stddev) ** 2))

    def cdf(self, x):
        """
        cdf (Cumulative Distribution Function) for a Normal Distribution
        """
        π = 3.1415926536
        xx = (x - self.mean) / (self.stddev * (2 ** (1 / 2)))
        erf = (2 / (π ** (1 / 2))) * (xx - (xx ** 3) / 3 + (xx ** 5) / 10
                                      - (xx ** 7) / 42 + (xx ** 9) / 216)
        return (1 + erf) / 2
