import tensorflow as tf
from tensorflow.contrib import rnn
from models.net_utils import get_init_embedding


class AttentionRNN(object):
    def __init__(self, reversed_dict, document_max_len, num_class, FLAGS, sequence_length, vocab_size, embedding_size, learning_rate = 1e-3):

              self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        self.num_hidden = args.num_hidden
        self.num_layers = args.num_layers
        self.learning_rate = learning_rate

        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.x_len = tf.reduce_sum(tf.sign(self.input_x), 1)
        self.input_y = tf.placeholder(tf.int32, [None], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, [], name="dropout_keep_prob")
        self.global_step = tf.Variable(0, trainable=False)

        with tf.name_scope("embedding"):
            if args.glove:
                init_embeddings = tf.constant(get_init_embedding(reversed_dict, embedding_size), dtype=tf.float32)
                self.embeddings = tf.get_variable("embeddings", initializer=init_embeddings, trainable=False)
            else:
                init_embeddings = tf.random_uniform([vocab_size, embedding_size])
                self.embeddings = tf.get_variable("embeddings", initializer=init_embeddings, trainable=True)
            self.data_embedding = tf.nn.embedding_lookup(self.embeddings, self.input_x)

        with tf.name_scope("birnn"):
            fw_cells = [rnn.BasicLSTMCell(self.num_hidden) for _ in range(self.num_layers)]
            bw_cells = [rnn.BasicLSTMCell(self.num_hidden) for _ in range(self.num_layers)]
            fw_cells = [rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob) for cell in fw_cells]
            bw_cells = [rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob) for cell in bw_cells]

            self.rnn_outputs, _, _ = rnn.stack_bidirectional_dynamic_rnn(
                fw_cells, bw_cells, self.data_embedding, sequence_length=self.x_len, dtype=tf.float32)

        with tf.name_scope("attention"):
            self.attention_score = tf.nn.softmax(tf.contrib.slim.fully_connected(self.rnn_outputs, 1))
            self.attention_out = tf.squeeze(
                tf.matmul(tf.transpose(self.rnn_outputs, perm=[0, 2, 1]), self.attention_score),
                axis=-1)

        with tf.name_scope("output"):
            self.logits = tf.contrib.slim.fully_connected(self.attention_out, num_classes, activation_fn=None)
            self.predictions = tf.argmax(self.logits, -1, output_type=tf.int32)

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y))
                
            opt = tf.train.AdamOptimizer(self.learning_rate)
            self.grads_and_vars = opt.compute_gradients(self.loss)
            self.optimizer = opt.apply_gradients(self.grads_and_vars, global_step=self.global_step)

            # This is the same as doing the following: 
            # self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
