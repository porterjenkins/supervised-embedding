#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Test the unsupervised embedding generate_batch function

@author: hxw186
"""

from unsupervised_embedding import generate_batch, prepare_data
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

    def test_prepare_data(self):
        d = [['a', 'b', 'e'], ['a', 'e'], ['w', 'b'], ['e']]
        ds, n, d_seq, seq_d = prepare_data(d)
        assert(n == 4)
        assert(seq_d[d_seq['a']] == 'a')
        assert(seq_d[d_seq['b']] == 'b')
        assert(seq_d[d_seq['e']] == 'e')
        assert(seq_d[d_seq['w']] == 'w')
        assert('c' not in d_seq)
        print ds 

        
if __name__ == '__main__':
    unittest.main()
