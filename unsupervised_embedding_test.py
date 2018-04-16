#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Test the unsupervised embedding generate_batch function

@author: hxw186
"""

from unsupervised_embedding import generate_batch
import unittest


class TestingFunc_generate_batch(unittest.TestCase):
    
    def test_generate_batch(self):
        batchSize = 4
        trips = [range(9), range(10, 30, 10), range(100, 310, 100)]
        b, l, ct, cr, n = generate_batch(trips, batchSize, 2, 1)
        print b, l.reshape(len(l)), ct, cr
        while n:
            b, l, ct, cr, n = generate_batch(trips, batchSize, 2, 1, ct, cr)
            print b, l.reshape(len(l)), ct, cr
        
if __name__ == '__main__':
    unittest.main()