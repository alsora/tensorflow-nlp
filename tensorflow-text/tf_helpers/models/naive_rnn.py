import tensorflow as tf
from tensorflow.contrib import rnn
from tf_helpers.net_utils import get_init_embedding


class NaiveRNN(object):
    def __init__(
        self, reversed_dict, sequence_length, num_classes,
        embedding_size, num_cells, num_layers, glove_vectors_dir = None, learning_rate = 1e-3):

        self.learning_rate = learning_rate

        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.x_len = tf.reduce_sum(tf.sign(self.input_x), 1)
        self.input_y = tf.placeholder(tf.int32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, [], name="dropout_keep_prob")
        self.global_step = tf.Variable(0, trainable=False)

        with tf.name_scope("embedding"):
            if glove_vectors_dir:
                init_embeddings = tf.constant(get_init_embedding(reversed_dict, embedding_size, glove_vectors), dtype=tf.float32)
                self.embeddings = tf.get_variable("embeddings", initializer=init_embeddings, trainable=False)
            else:
                vocab_size = len(reversed_dict)
                init_embeddings = tf.random_uniform([vocab_size, embedding_size])
                self.embeddings = tf.get_variable("embeddings", initializer=init_embeddings, trainable=True)

            self.data_embedding = tf.nn.embedding_lookup(self.embeddings, self.input_x)

        with tf.name_scope("birnn"):
            fw_cells = [rnn.BasicLSTMCell(num_cells) for _ in range(num_layers)]
            bw_cells = [rnn.BasicLSTMCell(num_cells) for _ in range(num_layers)]
            fw_cells = [rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob) for cell in fw_cells]
            bw_cells = [rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob) for cell in bw_cells]

            self.rnn_outputs, _, _ = rnn.stack_bidirectional_dynamic_rnn(
                fw_cells, bw_cells, self.data_embedding, sequence_length=self.x_len, dtype=tf.float32)
            self.last_output = self.rnn_outputs[:, -1, :]

        with tf.name_scope("output"):
            #self.logits = tf.contrib.slim.fully_connected(self.last_output, num_classes, activation_fn=None, name="logits")

            W = tf.get_variable("W", shape=[self.last_output.shape[1], num_classes], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.logits = tf.nn.xw_plus_b(self.last_output, W, b, name="logits")

            self.predictions = tf.argmax(self.logits, -1, name="predictions")

        with tf.name_scope("loss"):
            losses =  tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

            opt = tf.train.AdamOptimizer(self.learning_rate)
            self.grads_and_vars = opt.compute_gradients(self.loss)
            self.optimizer = opt.apply_gradients(self.grads_and_vars, global_step=self.global_step)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
