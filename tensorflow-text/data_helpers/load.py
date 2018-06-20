import numpy as np
import re
import itertools
import json
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
    return string.strip().lower()


def adapt_unique_file(single_data_file):
    examples = list(open(single_data_file, "r").readlines())

    examples = [[s[1:].strip(), s[0]] for s in examples]

    positive_examples= [x for x, y in examples if y == '1']
    negative_examples = [x for x, y in examples if y == '0']

    return positive_examples, negative_examples


def combine_data_files(data_files : list):
    lines = []
    for d in data_files:
        with open(d) as fin: lines.extend(fin.readlines())

    return lines

def load_data_and_labels(data_files : list):
    """
    Loads data from files,with two different file formattation, both with unique or divided files
    It splits the data into words and generates labels.
    Returns split sentences and labels.
    """
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
    for t in examples:
        t = t.strip()
        split = t.split('\t',1)
        labels.append(split[0])
        x_text.append(split[1])

    assert len(x_text) == len(labels)

    # count distinct labels
    distinct_labels = set(labels)
    num_labels = len(distinct_labels)

    dict_labels = {}

    for i,el in enumerate(distinct_labels):
        dict_labels[el] = i

    y = []

    for l in labels:
        one_hot_vect = np.zeros(num_labels,dtype=int).tolist()
        one_hot_vect[dict_labels[l]] = 1
        y.append(one_hot_vect)


    x_text = [clean_str(sent) for sent in x_text]

    with open('../dict_labels.json', 'w') as fp:
        json.dump(dict_labels, fp)

    return [x_text, y]




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




def build_dict(sentences, output_file):

    words = list()
    for sentence in sentences:
        for word in sentence.split(" "):
            words.append(word)

    word_counter = collections.Counter(words).most_common()
    word_dict = dict()
    word_dict["<padding>"] = 0
    word_dict["<unk>"] = 1
    for word, _ in word_counter:
        word_dict[word] = len(word_dict)


    # Save vocabulary to file
    print("Saving vocabulary to file: " + output_file)
    with open(output_file, "w") as f:
        f.write("\n".join(token for token in word_dict))
    print("- done.")


    reversed_dict = dict(zip(word_dict.values(), word_dict.keys()))

    return word_dict, reversed_dict



def transform_text(data, word_dict, max_element_length, padding="<padding>"):

    x = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), data))
    x = list(map(lambda d: d[:max_element_length], x))
    x = list(map(lambda d: d + (max_element_length - len(d)) * [word_dict[padding]], x))

    return x


if __name__ == "__main__":
    load_data_and_labels(["/home/mxm/tensorflow-text/example1.csv","/home/mxm/tensorflow-text/example2.csv"])