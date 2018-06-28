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

PADDING_ = "<padding>"
UNK_ = "<unk>"


def build_dict_words(sentences, output_dir = None, threshold_count = 1):

    words = list()
    for sentence in sentences:
        for word in sentence.split(" "):
            words.append(word)

    word_counter = collections.Counter(words).most_common()
    word_dict = dict()
    word_dict[PADDING_] = 0
    word_dict[UNK_] = 1
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

def build_dict_NER(sentences, output_dir = None, thresh_count = 1):

    words = list()
    for sentence in sentences:
        for word in sentence:
            words.append(word)

    word_counter = collections.Counter(words).most_common()
    word_dict = dict()
    word_dict[PADDING_] = 0
    word_dict[UNK_] = 1
    for word, count in word_counter:
        word_dict[word] = len(word_dict) if count > thresh_count else word_dict[UNK_]


    # Save vocabulary to file
    if output_dir:
        output_file = os.path.join(output_dir, "vocab_words")
        print("Saving vocabulary to file: " + output_file)
        with open(output_file, "w") as f:
            f.write("\n".join(token for token in word_dict))

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

    # Build vocabulary
    max_element_length = max([len(x.split(" ")) for x in data]) 
    # max_element_length = 20

    x = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict[UNK_]), d.split(" "))), data))
    x = list(map(lambda d: d[:max_element_length], x))
    x = list(map(lambda d: d + (max_element_length - len(d)) * [word_dict[PADDING_]], x))

    return x


def transform_text_NER(data,labels,labels_dict, word_dict, max_element_length):


    x = list(map(lambda d: list([word_dict.get(w, word_dict[UNK_]) for w in d]), data))
    x = list(map(lambda d: d[:max_element_length], x))
    x = list(map(lambda d: d + (max_element_length - len(d)) * [word_dict[PADDING_]], x))


    padding_label = labels_dict['O']
    padding_one_hot = np.zeros(len(labels_dict),dtype=int)
    padding_one_hot[labels_dict['O']] = 1

    y = list(map(lambda d: d + (max_element_length - len(d)) * [list(padding_one_hot)], labels))


    return x,y


def transform_labels(labels, labels_dict):

    # count distinct labels
    distinct_labels = set(labels)
    num_labels = len(distinct_labels)

    y = []
    for label in labels:
        one_hot_vect = [0] * num_labels
        one_hot_vect[labels_dict[label]] = 1
        y.append(one_hot_vect)

    return y

