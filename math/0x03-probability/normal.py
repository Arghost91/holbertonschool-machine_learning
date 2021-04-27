#!/usr/bin/env python3
"""

"""

class Normal:
    
    def __init__(self, data=None, mean=0., stddev=1.):

        self.mean = float(mean)
        self.stddev = float(stddev)
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
        else:
            self.mean = sum(data)/len(data)
            var = [(x  - self.mean)**2 for x in data]
            self.stddev = (sum(var)/len(data))**(1/2)
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
                
    def z_score(self, x):
        
        return (x - self.mean)/self.stddev
    
    def x_value(self, z):
        
        return (z*self.stddev) + self.mean
    
    def pdf(self, x):
        
        e = 2.7182818285
        π = 3.1415926536
        return (1/(self.stddev*(2π)**(1/2))*e**(-(1/2)*((x-self.mean)/self.stddev)**2)
