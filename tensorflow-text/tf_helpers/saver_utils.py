#! /usr/bin/env python

import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np
import os
import time


def frozen_graph_from_checkpoint(model_dir, output_node_names = ''):

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_folder + "/frozen_model.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

   # We import the meta graph and retrieve a Saver
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    
    # Before exporting our graph, we need to precise what is our output node
    if not output_node_names:
        output_node_names = "output/predictions"
    output_nodes_array = map(str.strip, output_node_names.split(","))

    # We start a session and restore the graph weights
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)

        output_graph_def = graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            input_graph_def, # The graph_def is used to retrieve the nodes
            output_nodes_array, # The output node names are used to select the useful nodes
            variable_names_blacklist=['global_step'])
    
    # Finally we serialize and dump the output graph to the filesystem
    with tf.gfile.GFile(output_graph, "wb") as output_graph_file:
        output_graph_file.write(output_graph_def.SerializeToString())

    print("%d ops in the final graph." % len(output_graph_def.node))
