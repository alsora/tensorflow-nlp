import tensorflow as tf
import argparse
import datetime
import os
import numpy as np
from tf_helpers.models import seq2seq
import data_helpers.vocab as vocab_utils
import data_helpers.load as load_utils
#from utils import build_dict, build_dataset, batch_iter



tf.flags.DEFINE_string("model", "seq2seq", "Network model to train: ner_lstm (default: ner_lstm)")


tf.flags.DEFINE_integer("num_hidden", 150, help="Network size.")
tf.flags.DEFINE_integer("num_layers",  2, help="Network depth.")
tf.flags.DEFINE_integer("beam_width",  10, help="Beam width for beam search decoder.")
tf.flags.DEFINE_string("glove_embedding", "", "Path to a file containing Glove pretrained vectors")


tf.flags.DEFINE_string("x_train_filepath", "../data/dataset/sample_seq2seq/train_small/train.article.txt", "Path to a file containing X training sentence")
tf.flags.DEFINE_string("y_train_filepath", "../data/dataset/sample_seq2seq/train_small/train.title.txt", "Path to a file containing Y training title")
tf.flags.DEFINE_string("x_dev_filepath", "../data/dataset/sample_seq2seq/train/valid.article.filter.txt", "Path to a file containing X valid sentence")
tf.flags.DEFINE_string("y_dev_filepath", "../data/dataset/sample_seq2seq/train/valid.title.filter.txt", "Path to a file containing Y valid title")


tf.flags.DEFINE_integer("summary_max_len", 15, "Max length of output summarizations")

tf.flags.DEFINE_string("output_dir", "", "Where to save the trained model, checkpoints and stats (default: current_dir/runs/timestamp)")

tf.flags.DEFINE_string("fasttext_embedding", "", "Path to a file containing Fasttext pretrained vectors")
tf.flags.DEFINE_integer("embedding_size", 300, "Dimensionality of character embedding")
tf.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate for backpropagation")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs")
tf.flags.DEFINE_float("dropout_keep_prob", 0.75, "Dropout keep probability")
tf.flags.DEFINE_boolean("small_data", False, help="Use only 20K samples of data")

tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps (default: 2000)")


FLAGS = tf.flags.FLAGS

def preprocess():


    x_text, y_text = load_utils.load_seq2seq(FLAGS.x_train_filepath, FLAGS.y_train_filepath)

    print("Building dictionary...")
    full_text = x_text + y_text
    word_dict, reversed_dict = vocab_utils.build_dict_words(full_text,"seq2seq", FLAGS.output_dir)


    print("Loading training dataset...")
    x_train = vocab_utils.transform_text(x_text, word_dict)
    y_train = vocab_utils.transform_text_y_seq2seq(y_text, word_dict, FLAGS.summary_max_len)

    x_train = np.array(x_train)
    y_train = np.array(y_train)


    return x_train, y_train, word_dict, reversed_dict, [], []




def train(x_train, y_train, word_dict, reversed_dict,  x_valid, y_valid):

    article_max_len = x_train.shape[1]

    with tf.Session() as sess:
        if (FLAGS.model == "seq2seq"):
            model = seq2seq.Seq2Seq(reversed_dict=reversed_dict,
                                    article_max_len=article_max_len,
                                    summary_max_len=FLAGS.summary_max_len,
                                    FLAGS=FLAGS)

        else:
            raise NotImplementedError()

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())

        batches = load_utils.batch_iter_seq2seq(x_train, y_train, FLAGS.batch_size, FLAGS.num_epochs)
        num_batches_per_epoch = (len(x_train) - 1) // FLAGS.batch_size + 1

        print("Iteration starts.")
        print("Number of batches per epoch :", num_batches_per_epoch)
        for batch_x, batch_y in batches:
            batch_x_len = list(map(lambda x: len([y for y in x if y != 0]), batch_x))
            batch_decoder_input = list(map(lambda x: [word_dict["<s>"]] + list(x), batch_y))
            batch_decoder_len = list(map(lambda x: len([y for y in x if y != 0]), batch_decoder_input))
            batch_decoder_output = list(map(lambda x: list(x) + [word_dict["</s>"]], batch_y))

            batch_decoder_input = list(
                map(lambda d: d + (FLAGS.summary_max_len - len(d)) * [word_dict["<padding>"]], batch_decoder_input))
            batch_decoder_output = list(
                map(lambda d: d + (FLAGS.summary_max_len - len(d)) * [word_dict["<padding>"]], batch_decoder_output))

            train_feed_dict = {
                model.batch_size: len(batch_x),
                model.input_x: batch_x,
                model.x_len: batch_x_len,
                model.decoder_input: batch_decoder_input,
                model.decoder_len: batch_decoder_len,
                model.input_y: batch_decoder_output
            }

            _, step, loss = sess.run([model.update, model.global_step, model.loss], feed_dict=train_feed_dict)

            if step % FLAGS.evaluate_every == 0:
                print("step {0}: loss = {1}".format(step, loss))

            if step % num_batches_per_epoch == 0:
                saver.save(sess, "./saved_model/model.ckpt", global_step=step)
                print("Epoch {0}: Model is saved.".format(step // num_batches_per_epoch))

def main(argv=None):

    if not FLAGS.output_dir:
        now = datetime.datetime.now()
        timestamp = str(now.strftime("%Y_%m_%d_%H_%M_%S"))
        FLAGS.output_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", FLAGS.model + timestamp))

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    x_train, y_train, word_dict, reversed_dict,  x_valid, y_valid = preprocess()
    train(x_train, y_train, word_dict, reversed_dict, x_valid, y_valid)

if __name__ == '__main__':
    tf.app.run()