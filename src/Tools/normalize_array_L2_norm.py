import numpy as np
from numpy import linalg as LA

#normalize per row
def normalize_array(array_X, axis=1):
    if axis==1:
        norm_array_X = LA.norm(array_X, axis=1)
        print('norm shape', norm_array_X.shape)
        print('array_X shape', array_X.shape)
        for row in range(array_X.shape[0]):
            for column in range(array_X.shape[1]):
                array_X[row, column] = array_X[row, column] / norm_array_X[row]
    elif axis==0:
        norm_array_X = LA.norm(array_X, axis=0)
        print('norm shape', norm_array_X.shape)
        print('array_X shape', array_X.shape)
        for column in range(array_X.shape[1]):
            for row in range(array_X.shape[0]):
                array_X[row, column] = array_X[row, column] / norm_array_X[column]
    return array_X

