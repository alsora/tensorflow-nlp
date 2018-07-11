#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers.load as load_utils
import data_helpers.vocab as vocab_utils
from tf_helpers.models import ner_lstm
from tf_helpers import saver_utils
from tensorflow.contrib import learn
from sklearn.model_selection import StratifiedShuffleSplit

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("data", "../data/dataset/sample_sequence_tagging/train.tsv", "Data source tab separated files. It's possible to provide more than 1 file using a comma")

# Network type
tf.flags.DEFINE_string("model", "ner_lstm", "Network model to train: ner_lstm (default: ner_lstm)")

# Model directory
tf.flags.DEFINE_string("output_dir", "", "Where to save the trained model, checkpoints and stats (default: current_dir/runs/timestamp)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 10)")

# Saver parameters
tf.flags.DEFINE_integer("evaluate_every", 2000, "Evaluate model on dev set after this many steps (default: 2000)")
tf.flags.DEFINE_integer("checkpoint_every", 2000, "Save model after this many steps (default: 2000)")
tf.flags.DEFINE_integer("num_checkpoints", 25, "Max number of checkpoints to store (default: 25)")
tf.flags.DEFINE_boolean("summary", False, "Save train summaries to folder (default: False)")

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
    x_text, y_text = load_utils.load_sequence_data_and_labels(files_list)

    # Build vocabulary
    max_element_length = max([len(e) for e in x_text])

    word_dict, reversed_dict = vocab_utils.build_dict_words(x_text, "sequence_tagging", FLAGS.output_dir)
    labels_dict, _ = vocab_utils.build_sequence_dict_labels(y_text, FLAGS.output_dir)

    x = vocab_utils.transform_text(x_text, word_dict)
    y = vocab_utils.transform_sequence_labels(y_text, labels_dict)

    x = np.array(x)
    y = np.array(y)

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_valid = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_valid = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x, y

    print("Vocabulary Size: {:d}".format(len(word_dict)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_valid)))

    return x_train, y_train, word_dict, reversed_dict, labels_dict, x_valid, y_valid




def train(x_train, y_train, word_dict, reversed_dict, labels_dict, x_valid, y_valid):
    # Training
    # ==================================================

    sequence_length = x_train.shape[1]
    num_classes = y_train.shape[-1]


    if (FLAGS.model == "ner_lstm"):
        model = ner_lstm.NER_LSTM(
            reversed_dict=reversed_dict,
            sequence_length=sequence_length,
            word_length=0,
            num_classes=num_classes,
            FLAGS=FLAGS)
    else:
        raise NotImplementedError()


    model.initialize_session()

    model.initialize_summaries()

    # Generate batches
    train_batches = load_utils.batch_iter(x_train, y_train, FLAGS.batch_size, FLAGS.num_epochs)
    num_batches_per_epoch = (len(x_train) - 1) // FLAGS.batch_size + 1

    max_accuracy = 0
    # Training loop. For each batch...
    for train_batch in train_batches:
        x_train_batch, y_train_batch = zip(*train_batch)
        loss = model.train_step(x_train_batch, y_train_batch)
        current_step = tf.train.global_step(model.session, model.global_step)

        if (current_step + 1) % 10 == 0:
            epoch = ( current_step // num_batches_per_epoch) + 1
            relative_step = (current_step % num_batches_per_epoch) + 1
            time_str = datetime.datetime.now().isoformat()
            print("{}: epoch {}/{}, step {}/{}, loss {:g}".format(time_str, epoch, FLAGS.num_epochs, relative_step, num_batches_per_epoch, loss))

        if current_step % FLAGS.evaluate_every == 0:

            valid_accuracy, cnf_matrix = model.valid_step(x_valid, y_valid)
            time_str = datetime.datetime.now().isoformat()
            print("Evaluation:")
            print("{}: valid_accuracy {:g}".format(time_str, valid_accuracy))
            print("Confusion matrix:")
            print(cnf_matrix)
        
            if valid_accuracy > max_accuracy:
                max_accuracy = valid_accuracy
                path = os.path.join(FLAGS.output_dir, "saved")
                saver_utils.save_model(model.session, path)
                print("Saved model with better accuracy to {}\n".format(path))

        if current_step % FLAGS.checkpoint_every == 0:
            model.save_session()



def main(argv=None):

    if not FLAGS.output_dir:
        now = datetime.datetime.now()
        timestamp = str(now.strftime("%Y_%m_%d_%H_%M_%S"))
        FLAGS.output_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", FLAGS.model + timestamp))

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    x_train, y_train, word_dict, reversed_dict, labels_dict, x_valid, y_valid = preprocess()
    train(x_train, y_train, word_dict, reversed_dict, labels_dict, x_valid, y_valid)


if __name__ == '__main__':
    tf.app.run()