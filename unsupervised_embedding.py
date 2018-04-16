#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Generate embedding with word2vec

@author: hxw186
"""

import collections
import random
import numpy as np
import tensorflow as tf
from trip import Trip



def generate_batch(trips, batch_size, num_skips, skip_window, cur_trip=0, cur_road=0):
    """
    trips:
        the list of trips (sequences)
    batch_size:
        the size of one training batch
    num_skips:
        number of times to reuse a road to generate a label
    skip_window:
        nearby roads considered as context, e.g. [skip_window target skip_window].
    cur_trip:
        the index of trip being processed.
    cur_road:
        the index of road segment being processed in current trip.
    """
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buf = collections.deque(maxlen=span)
    i = 0
    next_available = True
    if cur_road + span <= len(trips[cur_trip]):
        buf.extend(trips[cur_trip][cur_road:cur_road+span])
        next_in = cur_road + span
    else:
        buf.extend(trips[cur_trip][cur_road:])
        next_in = len(trips[cur_trip])
    while i < batch_size:
        target_idx = skip_window if len(buf) == span else min(len(buf)-1, skip_window)
        context_idxs = [r for r in range(len(buf)) if r != target_idx]
        context_to_use = random.sample(context_idxs, min(num_skips, len(context_idxs)))
        for context_idx in context_to_use:
            batch[i] = buf[target_idx]
            labels[i, 0] = buf[context_idx]
            i += 1
            if i == batch_size:
                break
        if next_in >= len(trips[cur_trip]): # buffer contains incomplete previous trips
            buf.popleft()
            cur_road += 1
            if len(buf) == 1:
                # add next trip to buffer
                buf.clear()
                cur_trip += 1
                if cur_trip == len(trips): # no next trips
                    cur_trip = len(trips) - 1
                    next_available = False
                cur_road = 0
                if cur_road + span <= len(trips[cur_trip]):
                    buf.extend(trips[cur_trip][cur_road:cur_road+span])
                    next_in = cur_road + span
                else:
                    buf.extend(trips[cur_trip][cur_road:])
                    next_in = len(trips[cur_trip])
        else: # further consume current trips
            buf.append(trips[cur_trip][next_in])
            cur_road += 1
            next_in += 1
    return batch, labels, cur_trip, cur_road, next_available
                    


if __name__ == "__main__":
    Trip.getTripsPickle()
    trips = [] # use road ID sequence to represent trips
    for t in Trip.all_trips:
        trips.append(t.trajectory["road_node"])

