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
            for x in data:
                var = (x  - self.mean)**2
            self.stddev = (var/len(data))**(1/2)
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif data < 2:
                raise ValueError("data must contain multiple values")
