import os
import urllib
import zipfile
import nltk
import numpy as np
import tensorflow as tf

def readGloveEmbeddings(pretrained_embedding=None):
    # I think ideally, our model would have generated a 300 dimentional dense
    # vector and we would just return the word closest to it but for some
    # reason this is not how things are done right now. Instead, words are
    # mapped to ids, ids are mapped to word vectors and the model contains a
    # final softmax layer which generates the probabilities for each word in
    # the vocabulary and this limits the vocab size!

    # We need a word2idx dictionary for mapping words to their index tokens -
    # we need to convert the words to indices to lookup their embeddings.
    # idx2word dictionary mapping the indices to words
    # embedding matrix of dim [vocab_size * embedding_dim] to store the word
    # embeddings.

    # The text file contains the word at the beginning followed by the
    # embedding values with spaces separating the values.
    with open(pretrained_embedding, 'r') as file:
        word2idx = {}
        idx2word = {}
        embedding_matrix = []
        for index,line in enumerate(file):
            values = line.split()
            word = values[0]
            word2idx[word] = index
            idx2word[index] = word
            word_embedding = np.asarray(values[1:],dtype=np.float32)
            embedding_matrix.append(word_embedding)
            if index > 40:
                break

def readDatafile(filename=None):
    with open(filename, 'r') as file:
        for line in file:
            print(line)

flags = tf.flags
flags.DEFINE_string('filename',None,'File containing the input data')
FLAGS = flags.FLAGS
print(FLAGS.filename)

readDatafile(FLAGS.filename)
