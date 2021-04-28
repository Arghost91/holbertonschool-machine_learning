#!/usr/bin/env python3

class Binomial:

    def __init__(self, data=None, n=1, p=0.5):
      
        self.n = int(n)
        self.p = float(p)
        
        if data is None:
            if n < 1:
                raise ValueError("n must be a positive value")
            elif p >= 1 or p <= 0:
                raise ValueError("p must be greater than 0 and less than 1")
        else:
            mean = sum(data) / len(data)
            var = sum([(x  - mean) ** 2 for x in data]) / len(data)
            self.p = 1 - (var / mean)
            self.n = int(round(mean / self.p))
            self.p = mean / self.n
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
