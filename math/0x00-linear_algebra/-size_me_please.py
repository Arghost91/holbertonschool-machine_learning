#!/usr/bin/env python3

def matrix_shape(matrix):
  if type(matrix[0][0]) is list:
    return[len(matrix), len(matrix[0]), len(matrix[0][0])]
  else:
    return[len(matrix), len(matrix[0])]
