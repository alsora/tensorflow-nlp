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

def load_data_and_labels_NER(filename, output_dir = None, processing_word = None, processing_tag=None):

    sentences = []
    labels = []
    niter = 0
    with open(filename) as f:
        words, tags = [], []
        for line in f:
            line = line.strip()
            if (len(line) == 0 or line.startswith("-DOCSTART-")):
                if len(words) != 0:
                    niter += 1
                    sentences.append(words)
                    labels.append(tags)
                    words = []
                    tags = []
            else:
                ls = line.split(' ')
                word, tag = ls[0], ls[-1]
                if processing_word is not None:
                    word = processing_word(word)
                if processing_tag is not None:
                    tag = processing_tag(tag)
                words += [word]
                tags += [tag]

        # count distinct labels
        distinct_labels = set()
        for line in labels:
            for tag in line:
                distinct_labels.add(tag)


        if 'O' not in distinct_labels:
            distinct_labels.add('O')

        num_labels = len(distinct_labels)


        dict_labels = {}
        for i, label in enumerate(distinct_labels):
            dict_labels[label] = i

        y = []
        for label in labels:
            single_line_labels = []
            for l in label:
                one_hot_vect = [0] * num_labels
                one_hot_vect[dict_labels[l]] = 1
                single_line_labels.append(one_hot_vect)

            y.append(single_line_labels)

        if output_dir:
            output_file = os.path.join(output_dir, "vocab_labels")
            with open(output_file, 'w') as fp:
                json.dump(dict_labels, fp)

        return sentences, y , dict_labels



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

def batch_iter_NER(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """


    data_x, data_y = zip(*data)

    data_x = np.array(data_x)
    data_y = np.array(data_y)

    assert len(data_x) == len(data_y)

    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(len(data)))
            shuffled_data_x = data_x[shuffle_indices]
            shuffled_data_y = data_y[shuffle_indices]
        else:
            shuffled_data_x = data_x
            shuffled_data_y = data_y

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(data))

            yield list(zip(shuffled_data_x[start_index:end_index], shuffled_data_y[start_index:end_index]))




