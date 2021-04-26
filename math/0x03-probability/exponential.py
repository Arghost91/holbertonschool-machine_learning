#!/usr/bin/env python3
"""

"""

class Exponential:
    def __init__(self, data=None, lambtha=1.):
    
    
        self.lambtha = float(lambtha)
        if data is None:
            if lambtha<0:
                raise ValueError("lambtha must be a positive value")
        else:
            self.lambtha = 1/(sum(data)/len(data))
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data)<2:
                raise ValueError("data must contain multiple values")         
    
    def pdf(self, x):
        
        e = 2.7182818285
        if x < 0:
            return 0
        else:
            return self.lamptha*(e**-(self.lamptha*x)
