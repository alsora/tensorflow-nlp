import tensorflow as tf
from tensorflow.contrib import rnn
from tf_helpers.layer_utils import *
from base_model import BaseModel


hyperparams = { "embedding_dim": 300,
                "num_cells": 100,
                "num_layers": 2,
                "learning_rate": 1e-3,
                "glove_embedding": '',
                "fasttext_embedding": '',
                "dropout_keep_prob": 0.75,
                'l2_reg_lambda': 0.0
                }


class AttentionRNN(BaseModel):
    """
    A Bidirection LSTM with attention for text classification.
    Uses an embedding layer, followed by bLSTM layers, attention layer and softmax layer.
    """
    def __init__(
        self, reversed_dict, sequence_length, num_classes, FLAGS):

        self.hyperparams = hyperparams

        super(AttentionRNN, self).__init__(FLAGS)

        # Placeholders for input, output and dropout
        self.sequence_length = tf.constant(sequence_length, dtype=tf.int32, shape=[], name="sequence_length")
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder_with_default(tf.zeros([1, num_classes], tf.int32), [None, num_classes], name="input_y")
        self.num_classes = num_classes
        self.dropout_keep_prob = tf.placeholder_with_default(1.0, [], name="dropout_keep_prob")
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.learning_rate = self.hyperparams['learning_rate']
        self.l2_loss = tf.constant(0.0)

        self.dataset = get_dataset(self.input_x, self.input_y, FLAGS)

        self.iterator = tf.data.Iterator.from_structure(self.dataset.output_types, self.dataset.output_shapes)
        
        self.dataset_init_op = self.iterator.make_initializer(self.dataset, name='dataset_init')

        self.x, self.y = self.iterator.get_next()
        self.x_len = tf.reduce_sum(tf.sign(self.x), 1)
        
        with tf.variable_scope("embedding"):
            embedded_x = add_word_embedding_layer(self, reversed_dict)

        with tf.variable_scope("birnn"):
            rnn_output = add_birnn_layer(self, embedded_x)

        with tf.variable_scope("attention"):
            attention_output = add_attention_layer(self, rnn_output)

        with tf.variable_scope("output"):
            logits = add_fully_connected_layer(self, attention_output)

            predictions = compute_predictions(self, logits)
       
        with tf.variable_scope("loss"):
            loss = compute_softmax_loss(self, logits, self.y, self.l2_loss)

            apply_backpropagation(self, loss, 'adam', self.learning_rate)


        with tf.variable_scope("accuracy"):
            # Convert 1 hot input into a dense vector
            dense_y = tf.argmax(self.y, 1, output_type=tf.int32)
            
            compute_accuracy(self, predictions, dense_y)

            compute_confusion_matrix(self, predictions, dense_y, self.num_classes)
       