#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Generate embedding with word2vec

@author: hxw186
"""

import collections
import random
import math
import numpy as np
import tensorflow as tf
from trip import Trip
from database_access import DatabaseAccess


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
    dao = DatabaseAccess(city='', data_dir="data")
    Trip.setDao(dao)
    Trip.getTripsPickle()
    trips = [] # use road ID sequence to represent trips
    for t in Trip.all_trips:
        trips.append(t.trajectory["road_node"])
    
    # calculate number of road segments (TODO: improve efficiency)
    num_segments = len(np.unique([t for t in trips]))
    
    # Build and train a skip-gram model
    batch_size = 64
    embedding_size = 64
    skip_window = 1
    num_skips = 2
    num_sampled = 32
    
    graph = tf.Graph()
    with graph.as_default():
        
        # input
        with tf.name_scope("inputs"):
            train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
            
        with tf.device("/cpu:0"):
            with tf.name_scope("embeddings"):
                embeddings = tf.Variable(
                        tf.random_uniform([num_segments, embedding_size], -1.0, 1.0))
                embed = tf.nn.embedding_lookup(embeddings, train_inputs)
                
            with tf.name_scope("weights"):
                nce_weights = tf.Variable(
                        tf.truncated_normal(
                                [num_segments, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
            with tf.name_scope("biases"):
                nce_biases = tf.Variable(tf.zeros(num_segments))
                
        with tf.name_scope("loss"):
            loss = tf.reduce_mean(
                    tf.nn.nce_loss(
                            weights=nce_weights,
                            biases=nce_biases,
                            labels=train_labels,
                            inputs=embed,
                            num_sampled=num_sampled,
                            num_classes=num_segments
                            ))
        tf.summary.scalar("loss", loss)
        
        with tf.name_scope("optimizer"):
            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        
        merged = tf.summary.merge_all()
        
        init = tf.global_variables_initializer()
        
        saver = tf.train.Saver()
    
    # Begin training
    
    num_steps = 10000
    with tf.Session(graph=graph) as session:
        writer = tf.summary.FileWriter("/tmp/unsupervised-embedding.log", session.graph)
        
        init.run()
        
        average_loss = 0
        cur_trip = 0
        cur_road = 0
        next_avaliable = True
        steps = 0
        while next_avaliable:
            batch_inputs, batch_labels, cur_trip, cur_road, next_avaliable  \
                    = generate_batch(trips, batch_size, num_skips, 
                                     skip_window, cur_trip, cur_road)
            steps += 1
                
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
            
            run_metadata = tf.RunMetadata()

            _, summary, los_val = session.run(
                    [optimizer, merged, loss],
                    feed_dic=feed_dict,
                    run_metadata=run_metadata)
            average_loss += loss_val

            writer.add_summary(summary, steps)

            if steps % 1000 == 0:
               average_loss /= 1000
               print("average loss at step", steps, ":", average_loss)
               average_loss = 0
        
        final_embeddings = normalized_embeddings.eval()
        
        saver.save(session, "model.ckpt")

        writer.close()
