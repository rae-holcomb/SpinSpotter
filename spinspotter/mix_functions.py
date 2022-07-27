import numpy as np

def add(y1, y2):
    '''Function to do addtion to ingredients.

    Args:
        y1 (:obj:`array`): numpy vector. The current ingredient.
        y2 (:obj:`array`): numpy vector. The new ingredient.

    Returns:
        :obj:`array`: y1 + y2.
    '''

    return np.add(y1, y2)

def multiply(y1, y2):
    '''Function to do mulplication to ingredients.

    Args:
        y1 (:obj:`array`): numpy vector. The current ingredient.
        y2 (:obj:`array`): numpy vector. The new ingredient.

    Returns:
        :obj:`array`: y1 * y2
    '''

    return np.multiply(y1, y2)

def subtract(y1, y2):
    '''Function to do subtraction to ingredients.

    Args:
        y1 (:obj:`array`): numpy vector. The current ingredient.
        y2 (:obj:`array`): numpy vector. The new ingredient.

    Returns:
        :obj:`array`: y1 - y2
    '''

    return np.subtract(y1, y2)

def divide(y1, y2):
    '''Function to do division to ingredients.

    Args:
        y1 (:obj:`array`): numpy vector. The current ingredient.
        y2 (:obj:`array`): numpy vector. The new ingredient.

    Returns:
        :obj:`array`: y1 / y2
    '''

    return np.divide(y1, y2)

def convolve(y1, y2):
    '''Function to do concolution to ingredients.

    Args:
        y1 (:obj:`array`): numpy vector. The current ingredient.
        y2 (:obj:`array`): numpy vector. The new ingredient.

    Returns:
       :obj:`array`: convolution of y1 and y2, using "same" method in np.convolve.
    '''
    
    return np.convolve(y1, y2, "same")