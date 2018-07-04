#! /usr/bin/env python

import numpy as np
import re
import json
import os


def is_number(s):
    """
    Checks wether the provided string is a number. Accepted: 1 | 1.0 | 1e-3 | 1,0 
    """
    s = s.replace(',', '.')
    try: 
        float(s)
        return True
    except ValueError:
        return False


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


def combine_data_files(data_files):
    lines = []
    for d in data_files:
        with open(d) as fin: lines.extend(fin.readlines())

    return lines


def load_cleaned_text(filepath):
    with open(filepath, "r") as f:
        list_ = list(map(lambda x: clean_str(x), f.readlines()))

    return list_


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


def load_sequence_data_and_labels(data_files, output_dir = None):

    # Load data from files
    if len(data_files) > 1:
        examples = combine_data_files(data_files=data_files)
    elif len(data_files) == 1:
        examples = list(open(data_files[0], "r").readlines())
    else:
        examples = []

    x_text = [] 
    y_text = []
    sentence = ''
    tags = []
    for line in examples:
        line = line.strip()
        if (len(line) == 0 or line.startswith("-DOCSTART-")):
            sentence = sentence.strip()
            if sentence:
                x_text.append(sentence)
                y_text.append(tags)
                sentence = ''
                tags = []
        else:
            splitted_line = line.split(' ')
            token = splitted_line[0]
            tag = splitted_line[-1]
            sentence += token + ' '
            tags += [tag]

    # not required as the input sequence should be already tokenized
    #x_text = [clean_str(sent) for sent in x_text]

    return x_text, y_text



def batch_iter(data_x, data_y, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """

    data_x = np.array(data_x)
    data_y = np.array(data_y)
    assert len(data_x) == len(data_y)

    len_data = len(data_y)

    num_batches_per_epoch = int(len_data/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(len_data))
            shuffled_data_x = data_x[shuffle_indices]
            shuffled_data_y = data_y[shuffle_indices]
        else:
            shuffled_data_x = data_x
            shuffled_data_y = data_y

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len_data)

            yield list(zip(shuffled_data_x[start_index:end_index], shuffled_data_y[start_index:end_index]))



def batch_iter_seq2seq(inputs, outputs, batch_size, num_epochs):
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index], outputs[start_index:end_index]

