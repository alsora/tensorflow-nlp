import tensorflow as tf
import numpy as np
from tf_helpers.layer_utils import *
from base_model import BaseModel


hyperparams = { "embedding_dim": 128,
                "filter_sizes": "3,4,5",
                "num_filters": 128,
                "learning_rate": 1e-3,
                "glove_embedding": '',
                "fasttext_embedding": '',
                "dropout_keep_prob": 0.5,
                'l2_reg_lambda': 0.0
                }


class TextCNN(BaseModel):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, reversed_dict, sequence_length, num_classes, FLAGS):

        self.hyperparams = hyperparams

        super(TextCNN, self).__init__(FLAGS)

        # Placeholders for input, output and dropout
        self.sequence_length = tf.constant(sequence_length, dtype=tf.int32, shape=[], name="sequence_length")
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None, num_classes], name="input_y")
        self.num_classes = num_classes
        self.dropout_keep_prob = tf.placeholder_with_default(1.0, [], name="dropout_keep_prob")
        self.learning_rate = FLAGS.learning_rate
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.dataset = get_dataset(self.input_x, self.input_y, FLAGS)

        self.iterator = tf.data.Iterator.from_structure(self.dataset.output_types, self.dataset.output_shapes)
        
        self.dataset_init_op = self.iterator.make_initializer(self.dataset, name='dataset_init')

        self.x, self.y = self.iterator.get_next()

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.variable_scope("embedding"):
            vocab_size = len(reversed_dict)
            self.W = tf.Variable(tf.random_uniform([vocab_size, FLAGS.embedding_dim], -1.0, 1.0), name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.x)
            self.data_embedding = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(FLAGS.filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, FLAGS.embedding_dim, 1, FLAGS.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[FLAGS.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.data_embedding,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.variable_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.variable_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, self.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="logits")
            self.predictions = tf.argmax(self.logits, 1,output_type=tf.int32, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.variable_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y)
            self.loss = tf.add(tf.reduce_mean(losses), self.hyperparams["l2_reg_lambda"] * l2_loss, name="loss")

            opt = tf.train.AdamOptimizer(self.learning_rate)
            self.grads_and_vars = opt.compute_gradients(self.loss)
            self.optimizer = opt.apply_gradients(self.grads_and_vars, global_step=self.global_step, name="optimizer")


        # Accuracy
        with tf.variable_scope("accuracy"):
            # Convert 1 hot input into a dense vector
            dense_y = tf.argmax(self.y, 1, output_type=tf.int32)

            # Compute accuracy
            correct_predictions = tf.equal(self.predictions, dense_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

            # Compute a per-batch confusion matrix
            batch_confusion = tf.confusion_matrix(labels=dense_y, predictions=self.predictions, num_classes=self.num_classes)
            # Create an accumulator variable to hold the counts
            self.confusion = tf.get_variable('confusion', shape=[self.num_classes,self.num_classes], dtype=tf.int32, initializer=tf.zeros_initializer())
            # Create the update op for doing a "+=" accumulation on the batch
            self.confusion_update = tf.assign( self.confusion, self.confusion + batch_confusion, name='confusion_update')
