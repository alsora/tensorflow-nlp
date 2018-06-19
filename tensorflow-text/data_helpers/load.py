import numpy as np
import re
import itertools
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




def load_data_and_labels(positive_data_file, negative_data_file=None):
    """
    Loads data from files,with two different file formattation, both with unique or divided files
    It splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files


    if negative_data_file:
        positive_examples = list(open(positive_data_file, "r").readlines())
        positive_examples = [s.strip() for s in positive_examples]
        negative_examples = list(open(negative_data_file, "r").readlines())
        negative_examples = [s.strip() for s in negative_examples]
    else:
        positive_examples, negative_examples = adapt_unique_file(positive_data_file)

    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)

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
