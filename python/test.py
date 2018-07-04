#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
from tf_helpers.models import base_model
import data_helpers.load as load_utils
import data_helpers.vocab as vocab_utils
import data_helpers.evaluation as evaluation_utils 
import csv

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("data", "../data/dataset/sample_text_classification/test.tsv", "Data source tab separated files. It's possible to provide more than 1 file using a comma")
tf.flags.DEFINE_string("it", "", "Interactive mode for evaluating sentences from command line")
tf.flags.DEFINE_string("output_dir", "", "Directory containing a trained model, i.e. checkpoints, saved, vocab_words")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS


def loadModel():

    model = base_model.BaseModel(FLAGS)
    model.initialize_session()
    model.restore_saved_model(FLAGS.output_dir)

    return model



def evaluate(model):

    # Load data
    print("Loading data...")
    files_list = FLAGS.data.split(",")
    x_text, y_text = load_utils.load_data_and_labels(files_list)

    # Map data into vocabulary
    words_dict_path = os.path.join(FLAGS.output_dir, "vocab_words")
    labels_dict_path = os.path.join(FLAGS.output_dir, "vocab_labels")

    word_dict = vocab_utils.load_dict(words_dict_path)
    labels_dict = vocab_utils.load_dict(labels_dict_path)
    reversed_labels_dict = vocab_utils.reverse_dict(labels_dict)

    max_element_length = 119
    x = vocab_utils.transform_text_v2(x_text, word_dict, max_element_length)
    y = vocab_utils.transform_labels(y_text, labels_dict)

    x = np.array(x)
    y = np.array(y)

    print("\nEvaluating...\n")
    # Generate batches for one epoch
    batches = load_utils.batch_iter(x, y, FLAGS.batch_size, 1, shuffle=False)

    # Collect the predictions here
    all_predictions = []
    all_predictions = np.array(all_predictions)

    correct_predictions = 0
    total_predictions = 0
    total_correct = 0

    for batch in batches:
        x_batch, y_batch = zip(*batch)
        batch_predictions = model.predict_step(x_batch)
        #all_predictions = np.concatenate(list(batch_predictions, all_predictions))

        #batch_correct = float(sum(all_predictions == y_batch))
        #correct_predictions += batch_correct
        #total_predictions += len(x_batch)



def interactive(model):
    """Creates interactive shell to play with model
    Args:
        model: instance of network model
    """
    model.logger.info(
    "This is an interactive mode.\nTo exit, enter 'exit'.\nYou can enter a text like\ninput> I love Paris")

    words_dict_path = os.path.join(FLAGS.output_dir, "vocab_words")
    labels_dict_path = os.path.join(FLAGS.output_dir, "vocab_labels")

    word_dict = vocab_utils.load_dict(words_dict_path)
    reversed_labels_dict = vocab_utils.load_reverse_dict(labels_dict_path)

    while True:
        try:
            # for python 2
            text = raw_input("input> ")
        except NameError:
            # for python 3
            text = input("input> ")

        tokens = text.strip()

        if tokens == "exit":
            break

        data = [tokens]

        max_element_length = 119
        x = vocab_utils.transform_text_v2(data, word_dict, max_element_length)

        preds_ids = model.predict_step(x)
        preds = [reversed_labels_dict[idx] for idx in list(preds_ids[0])]

        to_print = evaluation_utils.align_data({"input": data, "output": preds})

        for key, seq in to_print.items():
            model.logger.info(seq)




def main(argv=None):

    print "FLAGS:", FLAGS.output_dir

    model = loadModel()

    if FLAGS.it or not FLAGS.data:
        interactive(model)
    else:
        evaluate(model)



if __name__ == '__main__':
    tf.app.run()