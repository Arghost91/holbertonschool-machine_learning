#!/usr/bin/env python3
"""
Class that represents a Poisson distribution
"""

class Poisson:
  """
  Class for Poisson
  """
  e = 2.7182818285
  
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
      
      if type(k) is not int:
          k = int(k)
      elif k<0:
          return 0
      return ((self.lambthaa**k)*(Poisson.e**-self.lambtha))/self.factorial(k)
