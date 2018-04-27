#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Generate embedding with word2vec

@author: hxw186
"""

import collections
import pickle
import random
import math
import numpy as np
import tensorflow as tf
import argparse 
from trip import Trip
from database_access import DatabaseAccess
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import init_ops

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
            if len(buf) > 1:
                buf.popleft()
                cur_road += 1
            if len(buf) <= 1:
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


def prepare_data(trips):
    _single_list = [r for t in trips for r in t]
    road_counts = dict(collections.Counter(_single_list))
    roadID_to_seqID = dict()
    for road, cnt in road_counts.items():
        roadID_to_seqID[road] = len(roadID_to_seqID)
    
    trips_new = []
    for trip in trips:
        trip_new = []
        for road_id in trip:
            trip_new.append(roadID_to_seqID[road_id])
        trips_new.append(trip_new)
    
    seqID_to_roadID = dict(zip(roadID_to_seqID.values(), roadID_to_seqID.keys()))
    return trips_new, len(roadID_to_seqID), roadID_to_seqID, seqID_to_roadID


def execute_w2v(dao,batch_size,embedding_size,skip_window,num_skips,num_sampled,num_steps,trips, num_segments, road_seq, seq_road):

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
        tf.summary.histogram("wegiths", nce_weights)
        tf.summary.histogram("biases", nce_biases)

        global_step = variable_scope.get_variable(  # this needs to be defined for tf.contrib.layers.optimize_loss()
            "global_step", [],
            trainable=False,
            dtype=dtypes.int64,
            initializer=init_ops.constant_initializer(0, dtype=dtypes.int64))

        with tf.name_scope("optimizer"):
            optimizer = tf.contrib.layers.optimize_loss(
                loss, global_step, learning_rate=0.001, optimizer="Adam",
                summaries=["gradients"])

        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm

        tf.summary.histogram("embedding_norms", norm)

        merged = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

    # Begin training


    with tf.Session(graph=graph) as session:
        writer = tf.summary.FileWriter("/tmp/unsupervised-embedding", session.graph)

        init.run()

        average_loss = 0
        cur_trip = 0
        cur_road = 0
        next_avaliable = True
        steps = 0
        while next_avaliable:
            batch_inputs, batch_labels, cur_trip, cur_road, next_avaliable \
                = generate_batch(trips, batch_size, num_skips,
                                 skip_window, cur_trip, cur_road)
            steps += 1

            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            run_metadata = tf.RunMetadata()

            _, summary, loss_val = session.run(
                [optimizer, merged, loss],
                feed_dict=feed_dict,
                run_metadata=run_metadata)
            average_loss += loss_val

            writer.add_summary(summary, steps)

            if steps % num_steps == 0:
                average_loss /= num_steps
                print("average loss at step", steps, ":", average_loss)
                average_loss = 0

        final_embeddings = normalized_embeddings.eval()

        saver.save(session, "./model.ckpt")

        writer.close()

    with open(dao.data_dir + "road_embedding-window-{}.pickle".format(skip_window), "w") as fout:
        pickle.dump(final_embeddings, fout)
        pickle.dump(road_seq, fout)
        pickle.dump(seq_road, fout)


def main(dao,batch_size,embedding_size,skip_window,num_skips,num_sampled,num_steps):
    Trip.setDao(dao)
    Trip.getTripsPickle()
    trips_raw = []  # use road ID sequence to represent trips
    for t in Trip.all_trips:
        trips_raw.append(t.trajectory["road_node"].apply(
            lambda x: -1 if np.isnan(x) else int(x)).tolist())
        # missing road segments is mapped to -1

    trips, num_segments, road_seq, seq_road = prepare_data(trips_raw)

    # Build and train a skip-gram model
    execute_w2v(dao,batch_size, embedding_size, skip_window, num_skips, num_sampled, num_steps,trips, num_segments, road_seq, seq_road)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding parameter settings")
    parser.add_argument("--skip_window", nargs="?", const=1)

    arg = parser.parse_args()

    dao = DatabaseAccess(city='', data_dir="data")
    #dao = DatabaseAccess(city='jinan',
    #                     data_dir="/Volumes/Porter's Data/penn-state/data-sets/")


    
    # Build and train a skip-gram model
    batch_size = 32
    embedding_size = 100 
    skip_window = 1
    num_skips = 2
    num_sampled = 16
    num_steps = 5000

    main(dao,batch_size,embedding_size,skip_window,num_skips,num_sampled,num_steps)

