import numpy as np
import tensorflow as tf
import argparse
import sys
import os
from random import randint

from model import ModelV1

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}, to remove tf console output

def load_from_h5(filename):
    model = tf.keras.models.load_model(filename)
    model.summary()

    return model

def load_from_checkpoint(checkpoint_dir, vocab_size):
    model = ModelV1().build_model(vocab_size)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([1, None]))

    return model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5', nargs='?', const=True, default=False,
                    help='use h5')
    parser.add_argument('--ckpt', nargs='?', const=True, default=False, 
                    help='use checkpoint')
    return parser

if __name__ == '__main__':
    h5_file = 'results/rapper.h5'
    checkpoint_dir = 'results'
    TEST_SET_RAW_FILENAME = 'data/test_set.txt'
    TEST_SET_FILENAME = 'data/encoded_test_set.txt'
    VOCABSET_FILENAME = 'data/vocabSet.txt'
    WORD2IDX_FILENAME = 'data/word2idx.npy'
    IDX2WORD_FILENAME = 'data/idx2word.npy'
    OUTPUT_SEQ_LENGTH = 100

    # model being used
    modelv1 = ModelV1()
    params = [modelv1.EMBEDDING_DIM, modelv1.RNN_UNITS, modelv1.SEQ_LENGTH]

    # user input (choose h5 or checkpoint)
    parser = parse_args()
    args = parser.parse_args()
    if args.ckpt == False and args.h5 == False:
        print('Choose valid model file (--h5 or --ckpt')
        sys.exit()

    # get test set
    testData = []
    with open(TEST_SET_FILENAME, 'r') as f:
        for song in f.read().splitlines():
            add = []
            for num in song.split():
                add.append(int(num))
            testData.append(add)

    rawTestData = []
    with open(TEST_SET_RAW_FILENAME, 'r') as f:
        for song in f.read().splitlines():
            rawTestData.append(song.split())
        

    # split lyrics into segments of len = seq_length
    print('Splitting test songs into sequences of seq_length...')
    sequences = []
    raw_sequences = []
    for song in testData:
        for i in range(params[2], len(song)):
            sequences.append(song[i - params[2] : i])
    for song in rawTestData:
        for i in range(params[2], len(song)):
            raw_sequences.append(song[i - params[2] : i])

    # get lookup tables
    word2idx = np.load('data/word2idx.npy').item()
    idx2word = np.load('data/idx2word.npy')
    idx_not_in_vocabSet = len(word2idx)

    # load model
    model = load_from_h5(h5_file) if args.h5 else load_from_checkpoint(checkpoint_dir, len(word2idx)+1)
    print('Model loaded.\n')

    # generate predictions
    inputs = []
    rawInputs = []
    for _ in range(5):
        idx = randint(0, len(sequences))
        inputs.append(sequences[idx])
        rawInputs.append(raw_sequences[idx])

    outputs = []
    for i in inputs:
        seed = np.array([i])
        result = ''

        for _ in range(params[2]):
            output = model.predict(seed)
            output[0][idx_not_in_vocabSet] = 0 # ignore value at idx_not_in_vocabSet
            y_class = np.argmax(output, axis=-1)
            y = idx2word[y_class]
            if y[0] == '|-|':
                result += '\n'
            else:
                result += y[0] + ' '
            seed[0,:-1] = seed[0,1:]
            seed[0,-1] = y_class

        outputs.append(result)

    for i, o in zip(rawInputs, outputs):
        print('\nINPUT LINES:')
        print(' '.join(i).replace(' |-| ', '\n'))
        print('\nPREDICTED NEXT LINES:')
        print(o)