import numpy as np
import re
import itertools
import json
import collections
import os
import operator


PADDING_ = "<padding>"
UNK_ = "<unk>"


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
    labels = []
    x_text = []
    for line in examples:
        line = line.strip()
        split = line.split('\t',1)
        labels.append(split[0])
        x_text.append(split[1])

    assert len(x_text) == len(labels)

    # count distinct labels
    distinct_labels = set(labels)
    num_labels = len(distinct_labels)

    dict_labels = {}
    for i,label in enumerate(distinct_labels):
        dict_labels[label] = i

    y = []
    for label in labels:
        one_hot_vect = [0] * num_labels
        one_hot_vect[dict_labels[label]] = 1
        y.append(one_hot_vect)

    x_text = [clean_str(sent) for sent in x_text]

    if output_dir:
        output_file = os.path.join(output_dir, "vocab_labels")
        with open(output_file, 'w') as fp:
            json.dump(dict_labels, fp)

    return x_text, y




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




def build_dict(sentences, output_dir = None, thresh_count = 1):

    words = list()
    for sentence in sentences:
        for word in sentence.split(" "):
            words.append(word)

    word_counter = collections.Counter(words).most_common()
    word_dict = dict()
    word_dict[PADDING_] = 0
    word_dict[UNK_] = 1
    for word, count in word_counter:
        if count >= thresh_count:
            word_dict[word] = len(word_dict)
        else:
            word_dict[word] = word_dict[UNK_]        

    # Save vocabulary to file
    if output_dir:
        output_file = os.path.join(output_dir, "vocab_words")
        print("Saving vocabulary to file: " + output_file)
        sorted_dict = sorted(word_dict.items(), key=operator.itemgetter(1))
        with open(output_file, "w") as f:
            f.write("\n".join(elem[0] for elem in sorted_dict))

    reversed_dict = dict(zip(word_dict.values(), word_dict.keys()))

    return word_dict, reversed_dict


def load_dict(path):

    word_dict = dict()
    with open(path) as f:
        for index, line in enumerate(f):
            if not line:
                print ("Error reading line from dict: " + line)
                return {}

            word_dict[line.strip()] = len(word_dict)


    return word_dict



def transform_text(data, word_dict, max_element_length):

    x = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict[UNK_]), d.split(" "))), data))
    x = list(map(lambda d: d[:max_element_length], x))
    x = list(map(lambda d: d + (max_element_length - len(d)) * [word_dict[PADDING_]], x))

    return x
