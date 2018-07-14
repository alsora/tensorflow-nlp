import tensorflow as tf
from tensorflow.contrib import rnn
from tf_helpers.layer_utils import *
from base_model import BaseModel


hyperparams = { "embedding_dim": 300,
                "embedding_dim_char": 100,
                "num_cells": 100,
                "num_layers": 2,
                "learning_rate": 1e-3,
                "glove_embedding": '',
                "fasttext_embedding": '',
                "dropout_keep_prob": 0.75,
                'l2_reg_lambda': 0.0,
                'use_crf': False,
                'use_chars': False
                }


class NER_LSTM(BaseModel):
    """
    A LSTM + CRF for NER.
    Uses an embedding layer, followed by bLSTM and NER.
    """
    #self, reversed_dict, sequence_length, num_classes, FLAGS):
    def __init__( self, reversed_dict, sequence_length, word_length, num_classes, FLAGS):

        self.hyperparams = hyperparams

        super(NER_LSTM, self).__init__(FLAGS)

        # Placeholders for input, output and dropout
        self.sequence_length = tf.constant(sequence_length, dtype=tf.int32, shape=[], name="sequence_length")
        self.input_x = tf.placeholder(tf.int32, shape=[None, sequence_length], name="input_x")
        self.input_x_char = tf.placeholder(tf.int32, shape=[None, sequence_length, word_length], name="input_x_char")
        self.input_y = tf.placeholder_with_default(tf.zeros([1, sequence_length, num_classes], tf.int32), [None, sequence_length, num_classes], name="input_y")

        self.num_classes = num_classes
        # hyper parameters
        self.dropout_keep_prob = tf.placeholder_with_default(1.0, [], name="dropout_keep_prob")
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.learning_rate = self.hyperparams['learning_rate']

        self.dataset = get_dataset(self.input_x, self.input_y, FLAGS)

        self.iterator = tf.data.Iterator.from_structure(self.dataset.output_types, self.dataset.output_shapes)
        
        self.dataset_init_op = self.iterator.make_initializer(self.dataset, name='dataset_init')

        self.x, self.y = self.iterator.get_next()
        self.x_len = tf.reduce_sum(tf.sign(self.x), 1)


        l2_loss = tf.constant(0.0)

        with tf.variable_scope("embedding"):
            if self.hyperparams['glove_embedding']:
                init_embeddings = tf.constant(get_glove_embedding(reversed_dict, self.hyperparams['glove_embedding']),dtype=tf.float32)
                self.embeddings = tf.get_variable("embeddings", initializer=init_embeddings, trainable=False)
            elif self.hyperparams['fasttext_embedding']:
                init_embeddings = tf.constant(get_fasttext_embedding(reversed_dict, self.hyperparams['fasttext_embedding']), dtype=tf.float32)
                self.embeddings = tf.get_variable("embeddings", initializer=init_embeddings, trainable=False)
            else:
                vocab_size = len(reversed_dict)
                init_embeddings = tf.random_uniform([vocab_size, self.hyperparams['embedding_dim']])
                self.embeddings = tf.get_variable("embeddings", initializer=init_embeddings, trainable=True)

            self.data_embedding = tf.nn.embedding_lookup(self.embeddings, self.x, name="word_embeddings")


        with tf.variable_scope("embedding_char"):
            if self.hyperparams['use_chars']:
                # get char embeddings matrix
                _char_embeddings = tf.get_variable(name="_char_embeddings", dtype=tf.float32, shape=[FLAGS.nchar, FLAGS.dim_char])

                char_embeddings = tf.nn.embedding_lookup(_char_embeddings, self.char_ids, name="char_embeddings")

                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings, shape=[s[0] * s[1], s[-2], FLAGS.dim_char])

                word_lengths = tf.reshape(self.word_lengths, shape=[s[0] * s[1]])

                # bi lstm on chars
                cell_fw = tf.contrib.rnn.LSTMCell(FLAGS.hidden_size_char, state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(FLAGS.hidden_size_char, state_is_tuple=True)

                _output = tf.nn.bidirectional_dynamic_rnn( cell_fw, cell_bw, char_embeddings, sequence_length=word_lengths, dtype=tf.float32)

                # read and concat output
                _, ((_, output_fw), (_, output_bw)) = _output
                output = tf.concat([output_fw, output_bw], axis=-1)

                # shape = (batch size, max sentence length, char hidden size)
                output = tf.reshape(output, shape=[s[0], s[1], 2 * FLAGS.hidden_size_char])

                self.data_embedding = tf.concat([self.data_embedding, output], axis=-1)

        # Dropout
        self.data_embedding = tf.nn.dropout(self.data_embedding, self.dropout_keep_prob)

        """
            For each word in each sentence of the batch, it corresponds to a vector
            of scores, of dimension equal to the number of tags.
        """

        with tf.variable_scope("birnn"):

            fw_cells = [rnn.BasicLSTMCell(self.hyperparams['num_cells']) for _ in range(self.hyperparams['num_layers'])]
            bw_cells = [rnn.BasicLSTMCell(self.hyperparams['num_cells']) for _ in range(self.hyperparams['num_layers'])]
            fw_cells = [rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob) for cell in fw_cells]
            bw_cells = [rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob) for cell in bw_cells]

            self.rnn_outputs, _, _ = rnn.stack_bidirectional_dynamic_rnn(fw_cells, bw_cells, self.data_embedding, sequence_length=self.x_len, dtype=tf.float32)
            self.last_output = self.rnn_outputs


        with tf.variable_scope("output"):

            W = tf.get_variable("W", shape=[self.last_output.shape[-1], self.num_classes], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")

            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            nsteps = tf.shape(self.last_output)[1]
            output = tf.reshape(self.last_output, [-1,self.last_output.shape[-1]])
            reshape_logits = tf.nn.xw_plus_b(output, W, b,name="reshape_logits")
            self.logits = tf.reshape(reshape_logits, [-1, nsteps, self.num_classes],name="logits")
            self.predictions = tf.argmax(self.logits, -1, output_type=tf.int32, name="predictions")


        with tf.variable_scope("loss"):

            """Defines the loss"""
            if self.hyperparams['use_crf']:
                dense_y = tf.argmax(self.y, -1, output_type=tf.int32)
                log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                    self.logits, dense_y, sequence_length)
                self.trans_params = trans_params  # need to evaluate it for decoding
                self.loss = tf.reduce_mean(-log_likelihood, name="loss")
            else:
                losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y)
                #mask = tf.sequence_mask(sequence_length)
                #losses = tf.boolean_mask(losses, mask)
                self.loss = tf.add(tf.reduce_mean(losses), self.hyperparams["l2_reg_lambda"] * l2_loss, name="loss")


                opt = tf.train.AdamOptimizer(self.learning_rate)
                self.grads_and_vars = opt.compute_gradients(self.loss)
                self.optimizer = opt.apply_gradients(self.grads_and_vars, global_step=self.global_step, name="optimizer")

        with tf.variable_scope("accuracy"):
            # Convert 1 hot input into a dense vector
            dense_y = tf.argmax(self.y, -1, output_type=tf.int32)

            # Compute accuracy
            correct_predictions = tf.equal(self.predictions, dense_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

            dense_y = tf.reshape(dense_y, [-1])
            predictions = tf.reshape(self.predictions ,[-1])

            # Compute a per-batch confusion matrix
            batch_confusion = tf.confusion_matrix(labels=dense_y, predictions=predictions,num_classes=self.num_classes)
            # Create an accumulator variable to hold the counts
            self.confusion = tf.get_variable('confusion', shape=[self.num_classes,self.num_classes], dtype=tf.int32, initializer=tf.zeros_initializer())
            # Create the update op for doing a "+=" accumulation on the batch
            self.confusion_update = tf.assign( self.confusion, self.confusion + batch_confusion, name='confusion_update')
