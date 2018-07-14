#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import datetime
import data_helpers.load as load_utils
import data_helpers.vocab as vocab_utils
from tf_helpers.models import naive_rnn, attention_rnn, text_cnn
from tf_helpers import saver_utils
from tensorflow.contrib import learn
from sklearn.model_selection import StratifiedShuffleSplit

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("data", "../data/dataset/sample_text_classification/train.tsv", "Data source tab separated files. It's possible to provide more than 1 file using a comma")

# Network type
tf.flags.DEFINE_string("model", "blstm", "Network model to train: blstm | blstm_att | cnn")

# Model directory
tf.flags.DEFINE_string("model_dir", "", "Where to save the trained model, checkpoints and stats (default: pwd/runs/timestamp)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs")

# Saver parameters
tf.flags.DEFINE_integer("evaluate_every", 2000, "Evaluate model on dev set after this many steps")
tf.flags.DEFINE_integer("checkpoint_every", 2000, "Save model after this many steps")
tf.flags.DEFINE_integer("num_checkpoints", 25, "Max number of checkpoints to store")
tf.flags.DEFINE_boolean("summary", False, "Save train summaries to folder")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

def preprocess():
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    files_list = FLAGS.data.split(",")
    x_text, y_text = load_utils.load_data_and_labels(files_list)

    word_dict, reversed_dict = vocab_utils.build_dict_words(x_text, "text_classification", FLAGS.model_dir)
    labels_dict, _ = vocab_utils.build_dict_labels(y_text, FLAGS.model_dir)

    x = vocab_utils.transform_text(x_text, word_dict)
    y = vocab_utils.transform_labels(y_text, labels_dict)

    x = np.array(x)
    y = np.array(y)

    # Randomly shuffle data
    sss = StratifiedShuffleSplit(n_splits=1, test_size=FLAGS.dev_sample_percentage, random_state=None)
    for train_index, valid_index in sss.split(x, y):
        x_train, x_valid =  x[train_index], x[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]

    del x, y

    print("Vocabulary Size: {:d}".format(len(word_dict)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_valid)))
    
    return x_train, y_train, word_dict, reversed_dict, x_valid, y_valid



def train(x_train, y_train, word_dict, reversed_dict, x_valid, y_valid):
    # Training
    # ==================================================

    sequence_length = x_train.shape[1]
    num_classes = y_train.shape[1]


    if (FLAGS.model == "blstm"):
        model = naive_rnn.NaiveRNN(
            reversed_dict=reversed_dict,
            sequence_length=sequence_length,
            num_classes=num_classes,
            FLAGS=FLAGS)
    elif (FLAGS.model == "blstm_att"):
        model = attention_rnn.AttentionRNN(
            reversed_dict=reversed_dict,
            sequence_length=sequence_length,
            num_classes=num_classes,
            FLAGS=FLAGS)
    elif (FLAGS.model == "cnn"):
        model = text_cnn.TextCNN(
            reversed_dict = reversed_dict,
            sequence_length=sequence_length,
            num_classes=num_classes,
            FLAGS=FLAGS)
    else:
        raise NotImplementedError()


    model.initialize_session()

    model.initialize_summaries()

    max_accuracy = 0
    for epoch in range(FLAGS.num_epochs):

        print "Epoch ", epoch + 1, "/", FLAGS.num_epochs

        model.train_step(x_train, y_train)

        model.save_session()

        valid_accuracy,_ = model.test_step(x_valid, y_valid)

        if valid_accuracy > max_accuracy:
            max_accuracy = valid_accuracy
            model.save_model()


def main(argv=None):

    if not FLAGS.model_dir:
        now = datetime.datetime.now()
        timestamp = str(now.strftime("%Y_%m_%d_%H_%M_%S"))
        FLAGS.model_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", FLAGS.model + timestamp))

    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)

    x_train, y_train, word_dict, reversed_dict, x_valid, y_valid = preprocess()
    train(x_train, y_train, word_dict, reversed_dict, x_valid, y_valid)

if __name__ == '__main__':
    tf.app.run()
