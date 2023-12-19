import unittest
from pressure_heatmapping import image_toolbox

import numpy as np


class Test_ImageProcessing(unittest.TestCase):
    '''DF creation: smoke (1), oneshot (2)
    
    Path to df defined in initialization,
    If folder is moved, most tests will fail.'''
    def test_one_shot_img_avg(self):
        """
        Pass a random set of arrays to image averaging function.
        If anything is returned, the test should pass. 

        """
        matrix_one = np.ones([3, 3], int)
        matrix_two = np.ones([3, 3], int)
        list_matrix_ones = []
        list_matrix_ones.append(matrix_one)
        list_matrix_ones.append(matrix_two)
        try:
            test_result_matrix = image_toolbox.img_avg(list_matrix_ones)
            test_result_value = test_result_matrix[1, 1]
            self.assertAlmostEqual(test_result_value, 1)
        except RuntimeError:
            self.assertRaises(RuntimeError)
