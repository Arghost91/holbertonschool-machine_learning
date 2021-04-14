#!/usr/bin/env python3

def matrix_transpose(matrix):
	
	matrix_P = [[matrix[i][j]for i in range(len(matrix))] for j in range(len(matrix[0]))]
	return matrix_P	
