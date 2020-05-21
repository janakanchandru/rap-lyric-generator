import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional

class ModelV1:
    def __init__(self):
        self.SEQ_LENGTH = 100
        self.BATCH_SIZE = 128
        self.EMBEDDING_DIM = 256
        self.RNN_UNITS = 512

    def build_model(self, vocab_size):
        model = tf.keras.Sequential([
            Embedding(vocab_size, self.EMBEDDING_DIM, input_length=self.SEQ_LENGTH),

            LSTM(self.RNN_UNITS, return_sequences=True, stateful=False, recurrent_initializer='glorot_uniform'),

            LSTM(self.RNN_UNITS, return_sequences=False, stateful=False, recurrent_initializer='glorot_uniform'),

            Dense(vocab_size, activation='softmax')
        ])
        model.summary()

        return model

class ModelV2:
    def __init__(self):
        self.SEQ_LENGTH = 100
        self.BATCH_SIZE = 128
        self.EMBEDDING_DIM = 256
        self.RNN_UNITS = 512

    def build_model(self, vocab_size):
        model = tf.keras.Sequential([
            Embedding(vocab_size, self.EMBEDDING_DIM, input_length=self.SEQ_LENGTH),

            Bidirectional(LSTM(self.RNN_UNITS, 
                                return_sequences=True,
                                stateful=False,
                                recurrent_initializer='glorot_uniform')),

            Bidirectional(LSTM(self.RNN_UNITS, 
                                return_sequences=False,
                                stateful=False,
                                recurrent_initializer='glorot_uniform')),

            Dense(vocab_size, activation='softmax')
        ])
        model.summary()

        return model