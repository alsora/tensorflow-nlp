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
    A LSTM + CRF for Sequence Tagging.
    Uses an embedding layer, followed by bLSTM and Fully Connected layer.
    """
    def __init__( self, reversed_dict, sequence_length, word_length, num_classes, FLAGS):

        self.hyperparams = hyperparams

        super(NER_LSTM, self).__init__(FLAGS)

        # Placeholders for input, output and dropout
        self.sequence_length = tf.constant(sequence_length, dtype=tf.int32, shape=[], name="sequence_length")
        self.input_x = tf.placeholder(tf.int32, shape=[None, sequence_length], name="input_x")
        self.input_x_char = tf.placeholder(tf.int32, shape=[None, sequence_length, word_length], name="input_x_char")
        self.input_y = tf.placeholder_with_default(tf.zeros([1, sequence_length, num_classes], tf.int32), [None, None, num_classes], name="input_y")
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

        '''
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
        '''
        
        # Dropout
        embedded_x = tf.nn.dropout(embedded_x, self.dropout_keep_prob)

        with tf.variable_scope("birnn"):
            rnn_output = add_birnn_layer(self, embedded_x)

        with tf.variable_scope("output"):
            nsteps = tf.shape(rnn_output)[1]
            output = tf.reshape(rnn_output, [-1,rnn_output.shape[-1]])

            logits = add_fully_connected_layer(self, output)

            logits = tf.reshape(logits, [-1, nsteps, self.num_classes],name="logits")
            self.logits = logits

            predictions = compute_predictions(self, logits)


        with tf.variable_scope("loss"):
            if self.hyperparams['use_crf']:
                dense_y = tf.argmax(self.y, -1, output_type=tf.int32)
                log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                    self.logits, dense_y, self.x_len)
                self.trans_params = trans_params  # need to evaluate it for decoding
                self.loss = tf.reduce_mean(-log_likelihood, name="loss")
            else:
                losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y)
                #WARNING: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
                mask = tf.sequence_mask(self.x_len)
                losses = tf.boolean_mask(losses, mask)
                self.loss = tf.add(tf.reduce_mean(losses), self.hyperparams["l2_reg_lambda"] * self.l2_loss, name="loss")


            apply_backpropagation(self, self.loss, 'adam', self.learning_rate)


        with tf.variable_scope("accuracy"):
            # Convert 1 hot input into a dense vector
            dense_y = tf.argmax(self.y, -1, output_type=tf.int32)

            accuracy = compute_accuracy(self, self.predictions, dense_y)

            y_1d = tf.reshape(dense_y, [-1])
            preds_1d = tf.reshape(self.predictions ,[-1])

            compute_confusion_matrix(self, preds_1d, y_1d, self.num_classes)
