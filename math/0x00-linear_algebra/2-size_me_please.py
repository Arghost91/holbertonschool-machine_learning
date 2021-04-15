#!/usr/bin/env python3

# Module that calculates the shape of a matrix


def matrix_shape(matrix):
  # With matrix as input, the function return the shape of a matrix  
    if type(matrix[0][0]) is list:
        return[len(matrix), len(matrix[0]), len(matrix[0][0])]
    else:
        return[len(matrix), len(matrix[0])]
