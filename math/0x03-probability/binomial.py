#!/usr/bin/env python3
"""
Class that represents a Poisson distribution
"""


class Binomial:
    """
    Class for Poisson
    """
    def __init__(self, data=None, n=1, p=0.5):
        """
        Class Constructor
        """
        self.n = int(n)
        self.p = float(p)
        if data is None:
            if n < 1:
                raise ValueError("n must be a positive value")
            elif p >= 1 or p <= 0:
                raise ValueError("p must be greater than 0 and less than 1")
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            var = sum([(x - mean) ** 2 for x in data]) / len(data)
            self.p = 1 - (var / mean)
            self.n = int(round(mean / self.p))
            self.p = mean / self.n

    def pmf(self, k):
        """
        pmf (Probability Mass Function) for a Binomial Distribution
        """
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        nfact = 1
        for i in range(1, self.n + 1):
            nfact *= i
        kfact = 1
        for i in range(1, k + 1):
            kfact *= i
        nkfact = 1
        for i in range(1, (self.n - k) + 1):
            nkfact *= i
        q = 1 - self.p
        return (nfact / (kfact * nkfact)) * (self.p ** k) * (q ** (self.n - k))

    def cdf(self, k):
        """
        cdf (Cumulative Distributive Function) for a Binomial Distribution
        """
        acum = 0
        for i in range(k + 1):
            acum += self.pmf(i)
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        return acum
