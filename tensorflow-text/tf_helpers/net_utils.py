from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.wrappers import FastText
import numpy as np


def get_glove_embedding(reversed_dict, glove_file):
    print("Loading Glove vectors...")
    word2vec_file = get_tmpfile("word2vec_format.vec")
    glove2word2vec(glove_file, word2vec_file)
    word_vectors = KeyedVectors.load_word2vec_format(word2vec_file)

    embedding_size = word_vectors.vector_size

    word_vec_list = list()
    for _, word in sorted(reversed_dict.items()):
        try:
            word_vec = word_vectors.word_vec(word)
        except KeyError:
            word_vec = np.zeros([embedding_size], dtype=np.float32)

        word_vec_list.append(word_vec)

    return np.array(word_vec_list)


def get_fasttext_embedding(reversed_dict, fasttext_file):
    model = FastText.load_fasttext_format(fasttext_file)

    embedding_size = model.vector_size

    word_vec_list = list()
    for _, word in sorted(reversed_dict.items()):
        try:
            word_vec = model[word]
        except KeyError:
            word_vec = np.zeros([embedding_size], dtype=np.float32)

        word_vec_list.append(word_vec)

    return np.array(word_vec_list)