import pandas as pd
import numpy as np
import os
import tensorflow as tf
from random import shuffle
from tqdm import tqdm

from model import ModelV1

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}, to remove tf console output

def split_input_target(chunk):
    input_seq = chunk[:-1]
    target = chunk[-1]
    return input_seq, target

def make_target_categorical(X, y):
    vocabSet_size = 7016 # unfortunatley this needs to be hardcoded, can't find a dynamic way
    y = tf.one_hot(y, vocabSet_size)
    return X, y

def convert_to_word(X, y_idx, idx_not_in_vocabSet, idx2word):
    word_X = ''
    word_y = ''

    for idx in X:
        if idx == idx_not_in_vocabSet:
            word_X += 'n/a '
        else:
            word_X += idx2word[idx] + ' '

    if y_idx == idx_not_in_vocabSet:
        word_y = 'n/a (word not in vocabSet)'
    else:
        word_y = idx2word[y_idx]

    return word_X, word_y

def preprocess(encoded_train_set_filename, vocab_set_filename, word2idx_filename, idx2word_filename, 
                seq_length, batch_size, num_examples_to_use=0):
    # get training data
    songData = []
    with open(encoded_train_set_filename, 'r') as f:
        for song in f.read().splitlines():
            song_split = song.split()
            add = []
            for num in song_split:
                add.append(int(num))
            songData.append(add)

    # load vocabSet
    vocabSet = []
    with open(vocab_set_filename, 'r') as f:
        vocabSet = f.read().splitlines()

    vocabSet_size = len(vocabSet) + 1
    idx_not_in_vocabSet = len(vocabSet)

    # load lookup tables
    word2idx = np.load('data/word2idx.npy', allow_pickle=True).item()
    idx2word = np.load('data/idx2word.npy', allow_pickle=True)

    # split lyrics into segments of len = seq_length
    print('Splitting songs into sequences of seq_length...')
    sequences = []
    for song in tqdm(songData):
        for i in range(seq_length+1, len(song)):
            sequences.append(song[i - (seq_length+1) : i])
    if num_examples_to_use:
        sequences = sequences[:num_examples_to_use]
    
    # create TF dataset, allows for large datasets without memory issues
    print('\nCreating TF dataset object out of sequences, this may take awhile...')
    print('MAKE SURE you input the correct vocabSet size in make_target_categorical(): ', vocabSet_size)
    sequences_dataset = tf.data.Dataset.from_tensor_slices(sequences)
    dataset = sequences_dataset.map(split_input_target)
    dataset = dataset.map(make_target_categorical)

    print('\nSample input and ouput: ')
    for i, o in dataset.take(1):
        X = i.numpy()
        y = o.numpy()
        y_idx = np.argmax(y, axis=-1)
        word_X, word_y = convert_to_word(X, y_idx, idx_not_in_vocabSet, idx2word)

        print('input: {}'.format(X))
        print('output shape: {}'.format(y.shape))
        print('output idx: {}'.format(y_idx))
        print('\ninput word sequence: {}'.format(word_X))
        print('output word: {}'.format(word_y))

    buffer_size = 10000
    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

    print('\nTotal # of training examples: ', len(sequences))

    print('\nDataset created')

    return vocabSet, word2idx, idx2word, dataset


if __name__ == '__main__':
    # Constants
    EPOCHS = 50
    # NUM_EXAMPLES_TO_USE = 100000 #if you want to limit the number of examples used for training
    TRAIN_SET_FILENAME = 'data/encoded_train_val_set.txt'
    VOCABSET_FILENAME = 'data/vocabSet.txt'
    WORD2IDX_FILENAME = 'data/word2idx.npy'
    IDX2WORD_FILENAME = 'data/idx2word.npy'

    # prepare checkpoint directory
    checkpoint_dir = 'results'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)
    

    # initate model
    modelv1 = ModelV1()
    params = [modelv1.EMBEDDING_DIM, modelv1.RNN_UNITS, modelv1.SEQ_LENGTH, modelv1.BATCH_SIZE]

    # prepare data
    vocabSet, word2idx, idx2word, dataset = preprocess(TRAIN_SET_FILENAME, 
                                                        VOCABSET_FILENAME,
                                                        WORD2IDX_FILENAME,
                                                        IDX2WORD_FILENAME,
                                                        params[2], 
                                                        params[3])
                                                        # NUM_EXAMPLES_TO_USE)

    # build model
    model = modelv1.build_model(len(vocabSet)+1)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # train model
    history = model.fit(dataset, batch_size=params[3], epochs=EPOCHS, callbacks=[checkpoint_callback])

    model.save('results/rapper.h5')
    