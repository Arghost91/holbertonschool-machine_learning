#!/usr/bin/env python3

#Module that returns the transpose of a 2D matrix

def matrix_transpose(matrix):
	
	# With matrix as input, the function return the transpose of a 2D matrix 
	
	matrix_P = [[matrix[i][j]for i in range(len(matrix))] for j in range(len(matrix[0]))]
	return matrix_P	
