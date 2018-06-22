import tensorflow as tf
from tensorflow.contrib import rnn
from tf_helpers.layer_utils import *


class NaiveRNN(object):
    """
    A Bidirection LSTM for text classification.
    Uses an embedding layer, followed by bLSTM and softmax layer.
    """
    def __init__(
        self, reversed_dict, sequence_length, num_classes, FLAGS):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None, num_classes], name="input_y")
        self.x_len = tf.reduce_sum(tf.sign(self.input_x), 1)
        self.dropout_keep_prob = tf.placeholder(tf.float32, [], name="dropout_keep_prob")
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.learning_rate = FLAGS.learning_rate

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        with tf.name_scope("embedding"):
            if FLAGS.glove_embedding:
                init_embeddings = tf.constant(get_glove_embedding(reversed_dict, FLAGS.glove_embedding), dtype=tf.float32)
                self.embeddings = tf.get_variable("embeddings", initializer=init_embeddings, trainable=False)
            elif FLAGS.fasttext_embedding:
                init_embeddings = tf.constant(get_fasttext_embedding(reversed_dict, FLAGS.fasttext_embedding), dtype=tf.float32)
                self.embeddings = tf.get_variable("embeddings", initializer=init_embeddings, trainable=False)
            else:
                vocab_size = len(reversed_dict)
                init_embeddings = tf.random_uniform([vocab_size, FLAGS.embedding_dim])
                self.embeddings = tf.get_variable("embeddings", initializer=init_embeddings, trainable=True)

            self.data_embedding = tf.nn.embedding_lookup(self.embeddings, self.input_x)

        with tf.name_scope("birnn"):
            fw_cells = [rnn.BasicLSTMCell(FLAGS.num_cells) for _ in range(FLAGS.num_layers)]
            bw_cells = [rnn.BasicLSTMCell(FLAGS.num_cells) for _ in range(FLAGS.num_layers)]
            fw_cells = [rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob) for cell in fw_cells]
            bw_cells = [rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob) for cell in bw_cells]

            self.rnn_outputs, _, _ = rnn.stack_bidirectional_dynamic_rnn(
                fw_cells, bw_cells, self.data_embedding, sequence_length=self.x_len, dtype=tf.float32)
            self.last_output = self.rnn_outputs[:, -1, :]

        with tf.name_scope("output"):
            #self.logits = tf.contrib.slim.fully_connected(self.last_output, num_classes, activation_fn=None)

            W = tf.get_variable("W", shape=[self.last_output.shape[1], num_classes], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            self.logits = tf.nn.xw_plus_b(self.last_output, W, b, name="logits")
            self.predictions = tf.argmax(self.logits, -1, output_type=tf.int32, name="predictions")

        with tf.name_scope("loss"):
            losses =  tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + FLAGS.l2_reg_lambda * l2_loss

            opt = tf.train.AdamOptimizer(self.learning_rate)
            self.grads_and_vars = opt.compute_gradients(self.loss)
            self.optimizer = opt.apply_gradients(self.grads_and_vars, global_step=self.global_step)

        with tf.name_scope("accuracy"):
            # Convert 1 hot input into a dense vector
            dense_y = tf.argmax(self.input_y, 1, output_type=tf.int32)

            # Compute accuracy
            correct_predictions = tf.equal(self.predictions, dense_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

            # Compute a per-batch confusion matrix
            batch_confusion = tf.confusion_matrix(labels=dense_y, predictions=self.predictions, num_classes=num_classes)
            # Create an accumulator variable to hold the counts
            self.confusion = tf.Variable( tf.zeros([num_classes,num_classes], dtype=tf.int32 ), name='confusion' )
            # Create the update op for doing a "+=" accumulation on the batch
            self.confusion_update = self.confusion.assign( self.confusion + batch_confusion )




