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

        self.hyperparams['filter_sizes'] = list(map(int, self.hyperparams['filter_sizes'].split(",")))
        self.sequence_length_int = sequence_length
        
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

        # Embedding layer
        with tf.device('/cpu:0'), tf.variable_scope("embedding"):
            vocab_size = len(reversed_dict)
            init_embeddings = tf.random_uniform([vocab_size, self.hyperparams['embedding_dim']])

            self.data_embedding = tf.nn.embedding_lookup(init_embeddings, self.x)
            #self.data_embedding = tf.expand_dims(self.data_embedding, -1)

        with tf.variable_scope("conv-maxpool"):
            h_pool_flat = add_1d_conv_layer(self, self.data_embedding)
    
        with tf.variable_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        with tf.variable_scope("output"):            
            logits = add_fully_connected_layer(self, self.h_drop)

            predictions = compute_predictions(self, logits)

        with tf.variable_scope("loss"):
            loss = compute_softmax_loss(self, logits, self.y, self.l2_loss)

            apply_backpropagation(self, loss, 'adam', self.learning_rate)

        with tf.variable_scope("accuracy"):
            # Convert 1 hot input into a dense vector
            dense_y = tf.argmax(self.y, 1, output_type=tf.int32)
            
            compute_accuracy(self, predictions, dense_y)

            compute_confusion_matrix(self, predictions, dense_y, self.num_classes)
