"""Build vocabularies of words and tags from datasets"""

import argparse
import tensorflow as tf
from collections import Counter
import json
import os
import sys
import re



parser = argparse.ArgumentParser()
parser.add_argument('--min_count', default=1, help="Minimum count for words in the dataset",
                    type=int)
parser.add_argument('--input', default='', help="Path to the input txt file")
parser.add_argument('--output', default='', help="Path where to save the output vocab file")


# Hyper parameters for the vocab
NUM_OOV_BUCKETS = 1 # number of buckets (= number of ids) for unknown words
PAD_WORD = '<pad>'
PAD_TAG = 'O'


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def save_vocab_to_txt_file(vocab, txt_path):
    """Writes one token per line, 0-based line id corresponds to the id of the token.

    Args:
        vocab: (iterable object) yields token
        txt_path: (stirng) path to vocab file
    """
    with open(txt_path, "w") as f:
        f.write("\n".join(token for token in vocab))


def save_dict_to_json(d, json_path):
    """Saves dict to json file

    Args:
        d: (dict)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)


def update_vocab_from_file(txt_path, vocab, clean=True):
    """Update word and tag vocabulary from dataset

    Args:
        txt_path: (string) path to file, one sentence per line
        vocab: (dict or Counter) with update method

    Returns:
        dataset_size: (int) number of elements in the dataset
    """
    with open(txt_path) as f:
        for i, line in enumerate(f):
            if clean:
                line = clean_str(line)
            vocab.update(line.split(' '))


    return i + 1

def update_vocab_from_list(data, vocab, clean=True):
    """Update word and tag vocabulary from dataset

    Args:
        data: (list) list of strings containing one sentence per line
        vocab: (dict or Counter) with update method

    Returns:
        dataset_size: (int) number of elements in the dataset
    """

    for i, line in enumerate(data):
        if clean:
            line = clean_str(line)
        vocab.update(line.split(' '))


    return i + 1



def build_vocab_from_file(input_file, output_file, clean=True, min_count = 1, pad_element = "<pad>"):

    # Build word vocab with train and test datasets
    print("Building " + input_file + " vocabulary...")
    words = Counter()
    size_vocab = update_vocab_from_file(input_file, words, clean)
    print("- done.")

    # Only keep most frequent tokens
    words = [tok for tok, count in words.items() if count >= min_count]

    # Add pad token if specified
    if pad_element and pad_element not in words:
        words.append(pad_element)

    # Save vocabularies to file
    print("Saving vocabularies to file " + output_file)
    save_vocab_to_txt_file(words, output_file)
    print("- done.")


    # Save datasets properties in json file
    sizes = {
        'train_size': size_vocab,
        'vocab_size': len(words) + NUM_OOV_BUCKETS,
        'pad_element': pad_element,
        'num_oov_buckets': NUM_OOV_BUCKETS
    }
    #save_dict_to_json(sizes, output_file + "stats")

     # Logging sizes
    to_print = "\n".join("- {}: {}".format(k, v) for k, v in sizes.items())
    print("Characteristics of the dataset:\n{}".format(to_print))

    lookup_table = tf.contrib.lookup.index_table_from_file(output_file, num_oov_buckets=NUM_OOV_BUCKETS)


def build_vocab_from_list(data, output_file, clean=True, min_count = 1, pad_element = "<pad>"):

    # Build word vocab with train and test datasets
    print("Building vocabulary...")
    words = Counter()
    size_vocab = update_vocab_from_list(data, words, clean)
    print("- done.")

    # Only keep most frequent tokens
    words = [tok for tok, count in words.items() if count >= min_count]

    # Add pad token if specified
    if pad_element and pad_element not in words:
        words.append(pad_element)

    # Save vocabularies to file
    print("Saving vocabulary to file: " + output_file)
    save_vocab_to_txt_file(words, output_file)
    print("- done.")


    # Save datasets properties in json file
    sizes = {
        'train_size': size_vocab,
        'vocab_size': len(words) + NUM_OOV_BUCKETS,
        'pad_element': pad_element,
        'num_oov_buckets': NUM_OOV_BUCKETS
    }
    #save_dict_to_json(sizes, output_file + "stats")

     # Logging sizes
    to_print = "\n".join("- {}: {}".format(k, v) for k, v in sizes.items())
    print("Characteristics of the dataset:\n{}".format(to_print))

    lookup_table = tf.contrib.lookup.index_table_from_file(output_file, num_oov_buckets=NUM_OOV_BUCKETS)





def transform_data(data, lookup_table):

    # Convert line into list of tokens, splitting by white space
    data = data.map(lambda string: tf.string_split([string]).values)

    # Lookup tokens to return their ids
    data = data.map(lambda tokens: (lookup_table.lookup(tokens), tf.size(tokens)))

    return data




if __name__ == '__main__':
    args = parser.parse_args()

    build_vocab(args.input, args.output, args.min_count)


