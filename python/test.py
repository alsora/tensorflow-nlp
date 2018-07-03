#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from tf_helpers.models import base_model
from tensorflow.contrib import learn
import data_helpers.load as load_utils
import data_helpers.vocab as vocab_utils
import csv

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("data", "../data/dataset/sample_data/test.tsv", "Data source tab separated files. It's possible to provide more than 1 file using a comma")
tf.flags.DEFINE_string("it", "", "Interactive mode for evaluating sentences from command line")
tf.flags.DEFINE_string("model_dir", "", "Directory containing a trained model, i.e. checkpoints, saved, vocab_words")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS



def loadModel():

    model = base_model.BaseModel(FLAGS)
    model.initialize_session()
    model.restore_saved_model(FLAGS.model_dir)

    return model



def evaluate(model):
    '''
    # CHANGE THIS: Load data. Load your own data here
    if FLAGS.eval_train:
        x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
        y_test = np.argmax(y_test, axis=1)
    else:
        x_raw = ["a masterpiece four years in the making", "everything is off."]
        y_test = [1, 0]


    # Load data
    print("Loading data...")
    files_list = FLAGS.data.split(",")
    x_text, y = load_utils.load_data_and_labels(files_list)

    # Build vocabulary
    max_element_length = max([len(x.split(" ")) for x in x_text]) 
    # max_element_length = 20

    word_dict, reversed_dict = load_utils.build_dict(x_text, FLAGS.output_dir)

    x = load_utils.transform_text(x_text, word_dict, max_element_length)

    x = np.array(x)
    y = np.array(y)
    '''


    x_text = ["super beautiful like it very much best love", "terrible sad shit fuck worst ruined"]
    y = [1, 0]

    # Map data into vocabulary
    words_dict_path = os.path.join(FLAGS.model_dir, "vocab_words")
    word_dict = vocab_utils.load_dict(words_dict_path)

    x = vocab_utils.transform_text(x_text, word_dict)

    x = np.array(x)
    y = np.array(y)

    print("\nEvaluating...\n")

    # Evaluation
    # ==================================================
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []

            for x_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

    # Print accuracy if y_test is defined
    if y is not None:
        correct_predictions = float(sum(all_predictions == y))
        print("Total number of test examples: {}".format(len(y)))
        print("Accuracy: {:g}".format(correct_predictions/float(len(y))))

    # Save the evaluation to a csv
    predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
    out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
    print("Saving evaluation to {0}".format(out_path))
    with open(out_path, 'w') as f:
        csv.writer(f).writerows(predictions_human_readable)



def interactive(model):
    """Creates interactive shell to play with model
    Args:
        model: instance of network model
    """
    model.logger.info("""This is an interactive mode.
    To exit, enter 'exit'.
    You can enter a text like
    input> I love Paris""")

    words_dict_path = os.path.join(FLAGS.model_dir, "vocab_words")
    labels_dict_path = os.path.join(FLAGS.model_dir, "vocab_labels")

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

        if tokens == ["exit"]:
            break

        PADDING_ = "<padding>"
        UNK_ = "<unk>"

        data = [tokens]
        max_element_length = 119

        x = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict[UNK_]), d.split(" "))), data))
        x = list(map(lambda d: d[:max_element_length], x))
        x = list(map(lambda d: d + (max_element_length - len(d)) * [word_dict[PADDING_]], x))

        preds_ids = model.predict_step(x)
        preds = [reversed_labels_dict[idx] for idx in list(pred_ids[0])]

        to_print = align_data({"input": tokens, "output": preds})

        for key, seq in to_print.items():
            model.logger.info(seq)




def main(argv=None):


    model = loadModel()

    if FLAGS.it:
        interactive(model)
    else:
        evaluate(model)



if __name__ == '__main__':
    tf.app.run()