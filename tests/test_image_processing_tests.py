import unittest
from pressure_heatmapping import image_toolbox


class Test_ImageProcessing(unittest.TestCase):
    '''DF creation: smoke (1), oneshot (2)
    
    Path to df defined in initialization,
    If folder is moved, most tests will fail.'''
    def test_smoke_img_avg(self):
        """
        Pass a random set of arrays to image averaging function.
        If anything is returned, the test should pass. 

        """
        try:
            image_toolbox.img_avg(5)
        except RuntimeError:
            self.assertRaises(RuntimeError)
