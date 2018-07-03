"""Build vocabularies of words and tags from datasets"""

import argparse
import tensorflow as tf
import json
import os
import numpy as np
import sys
import re
import operator
import collections
from nltk.tokenize import word_tokenize
import re
import pickle


PADDING_ = "<padding>"
UNK_ = "<unk>"
SOL_ = "<s>"
EOL_ = "</s>"

DEFAULT_TOKENS_DICT =  {"seq2seq" : [ PADDING_, UNK_,SOL_,EOL_], "sequence_tagging" : [ PADDING_, UNK_], "text_classification" : [ PADDING_, UNK_]}


def build_dict_words(sentences, task_type, output_dir = None, threshold_count = 1):

    words = list()
    for sentence in sentences:
        for word in sentence.split(' '):
            words.append(word)

    word_counter = collections.Counter(words).most_common()
    word_dict = dict()

    for t in DEFAULT_TOKENS_DICT[task_type]:
        word_dict[t] = len(word_dict)

    for word, count in word_counter:
        if count >= threshold_count:
            word_dict[word] = len(word_dict)
        else:
            word_dict[word] = word_dict[UNK_]        

    # Save vocabulary to file
    if output_dir:
        output_file = os.path.join(output_dir, "vocab_words")
        print("Saving words vocabulary to file: " + output_file)
        sorted_dict = sorted(word_dict.items(), key=operator.itemgetter(1))
        with open(output_file, "w") as f:
            f.write("\n".join(elem[0] for elem in sorted_dict))

    reversed_dict = dict(zip(word_dict.values(), word_dict.keys()))

    return word_dict, reversed_dict


def build_dict_labels(labels, output_dir = None):

    # count distinct labels
    distinct_labels = set(labels)

    labels_dict = {}
    for i,label in enumerate(distinct_labels):
        labels_dict[label] = i

    # Save vocabulary to file
    if output_dir:
        output_file = os.path.join(output_dir, "vocab_labels")
        print("Saving labels vocabulary to file: " + output_file)
        sorted_dict = sorted(labels_dict.items(), key=operator.itemgetter(1))
        with open(output_file, "w") as f:
            f.write("\n".join(elem[0] for elem in sorted_dict))

    reversed_dict = dict(zip(labels_dict.values(), labels_dict.keys()))

    return labels_dict, reversed_dict


def build_sequence_dict_labels(labels, output_dir = None):

    # count distinct labels
    distinct_labels = set()
    for line in labels:
        for tag in line:
            distinct_labels.add(tag)


    labels_dict = {}
    for i, label in enumerate(distinct_labels):
        labels_dict[label] = i

    # Save vocabulary to file
    if output_dir:
        output_file = os.path.join(output_dir, "vocab_labels")
        print("Saving labels vocabulary to file: " + output_file)
        sorted_dict = sorted(labels_dict.items(), key=operator.itemgetter(1))
        with open(output_file, "w") as f:
            f.write("\n".join(elem[0] for elem in sorted_dict))

    reversed_dict = dict(zip(labels_dict.values(), labels_dict.keys()))

    return labels_dict, reversed_dict


def load_dict(path):

    dict_ = dict()
    with open(path) as f:
        for index, line in enumerate(f):
            if not line:
                print ("Error reading line from dict: " + line)
                return {}

            dict_[line.strip()] = len(dict_)

    return dict_


def load_reverse_dict(path):

    dict_ = dict()
    with open(path) as f:
        for index, line in enumerate(f):
            if not line:
                print ("Error reading line from dict: " + line)
                return {}
            dict_[len(dict_)] = line.strip()

    return dict_


def transform_text(data, word_dict):

    max_element_length = max([len(x.split(" ")) for x in data])
    # max_element_length = 20

    x = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict[UNK_]), d.split(" "))), data))
    x = list(map(lambda d: d[:max_element_length], x))
    x = list(map(lambda d: d + (max_element_length - len(d)) * [word_dict[PADDING_]], x))
    return x





def transform_text_y_seq2seq(data, word_dict, summary_max_len):

    y = list(map(lambda d: word_tokenize(d), data))
    y = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), y))
    y = list(map(lambda d: d[:(summary_max_len - 1)], y))
    return y


def transform_labels(labels, labels_dict):

    # count distinct labels
    num_labels = len(labels_dict)

    y = []
    for label in labels:
        one_hot_vect = [0] * num_labels
        one_hot_vect[labels_dict[label]] = 1
        y.append(one_hot_vect)

    return y




def transform_sequence_labels(labels, labels_dict):

    max_element_length = max([len(line) for line in labels]) 
    # count distinct labels
    num_labels = len(labels_dict)

    y = []
    for sequence_labels in labels:
        sequence_y = []
        for label in sequence_labels:
            one_hot_vect = [0] * num_labels
            one_hot_vect[labels_dict[label]] = 1
            sequence_y.append(one_hot_vect)
        y.append(sequence_y)

    # add labels padding
    padding_label = [0] * num_labels
    padding_label[labels_dict['O']] = 1

    y = list(map(lambda d: d + (max_element_length - len(d)) * [list(padding_label)], y))


    return y






