import numpy as np
import re
import itertools
import json
import os
import operator
import collections


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
    string = re.sub(r"@[A-Za-z0-9]+", " ", string)  #For Twitter use: remove hashtags
    return string.strip().lower()


def adapt_unique_file(single_data_file):
    examples = list(open(single_data_file, "r").readlines())

    examples = [[s[1:].strip(), s[0]] for s in examples]

    positive_examples= [x for x, y in examples if y == '1']
    negative_examples = [x for x, y in examples if y == '0']

    return positive_examples, negative_examples


def combine_data_files(data_files):
    lines = []
    for d in data_files:
        with open(d) as fin: lines.extend(fin.readlines())

    return lines

def load_data_and_labels(data_files, output_dir = None):

    # Load data from files
    if len(data_files) > 1:
        examples = combine_data_files(data_files=data_files)
    elif len(data_files) == 1:
        examples = list(open(data_files[0], "r").readlines())
    else:
        examples = []

    # Save label of every example
    y_text = []
    x_text = []
    for line in examples:
        line = line.strip()
        split = line.split('\t',1)
        y_text.append(split[0])
        x_text.append(split[1])

    assert len(x_text) == len(y_text)

    x_text = [clean_str(sent) for sent in x_text]


    return x_text, y_text




def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(len(data)))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(data))
            yield shuffled_data[start_index:end_index]




