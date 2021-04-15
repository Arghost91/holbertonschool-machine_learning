#!/usr/bin/env python3

# Module that adds two arrays element-wise


def add_arrays(arr1, arr2):

    """With two arrays as input, the function return the sum 
       of the two arrays as new list if the two arrays have the same shape""" 

    if len(arr1) != len(arr2):
        return None 
    else:
        return [arr1[i] + arr2[i] for i in range(len(arr1))]
