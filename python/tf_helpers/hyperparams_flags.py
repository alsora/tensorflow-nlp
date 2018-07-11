#! /usr/bin/env python

import tensorflow as tf


tf.flags.DEFINE_integer("embedding_dim", None, "Dimensionality of words embedding")
tf.flags.DEFINE_integer("embedding_dim_char", None, "Dimensionality of characterss embedding")
tf.flags.DEFINE_string("filter_sizes", None, "Comma-separated filter sizes")
tf.flags.DEFINE_integer("num_filters", None, "Number of filters per filter size")
tf.flags.DEFINE_integer("num_cells", None, "Number of cells in each BLSTM layer")
tf.flags.DEFINE_integer("num_layers", None, "Number of BLSTM layers")
tf.flags.DEFINE_float("learning_rate", None, "Learning rate for backpropagation")
tf.flags.DEFINE_string("glove_embedding", None, "Path to a file containing Glove pretrained vectors")
tf.flags.DEFINE_string("fasttext_embedding", None, "Path to a file containing Fasttext pretrained vectors")
tf.flags.DEFINE_float("dropout_keep_prob", None, "Dropout keep probability")
tf.flags.DEFINE_float("l2_reg_lambda", None, "L2 regularization lambda")
tf.flags.DEFINE_boolean("use_crf", None, "Use CRF instead of Sofmax")
tf.flags.DEFINE_boolean("use_chars", None, "Extend word embedding with char embedding")