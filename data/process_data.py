import pandas as pd
import re
import math
import numpy as np
from tqdm import tqdm

FILENAME = 'songs.csv'

songs = pd.read_csv(FILENAME)
songs['Lyrics'] = songs['Lyrics'].astype('str')

# split dataset into train/val and test
songData = songs['Lyrics'].str.split()
songsAsList = list(songData)
total_num = len(songsAsList)

train_len = math.ceil(total_num * 0.6)
val_len = math.ceil(total_num * 0.2 )
test_len = total_num - (train_len + val_len)

train_val_set = songsAsList[:train_len+val_len]
test_set = songsAsList[train_len+val_len:]

with open('train_val_set.txt', 'w') as f:
    for song in train_val_set:
        f.writelines('%s ' % word for word in song)
        f.write('\n')
with open('test_set.txt', 'w') as f:
    for song in test_set:
        f.writelines('%s ' % word for word in song)
        f.write('\n')

# create vocab set and get some metrics
vocabDict = {}
wordCount = 0
for song in train_val_set:
    for lyric in song:
        if lyric in vocabDict:
            vocabDict[lyric] += 1
        else:
            vocabDict[lyric] = 1
        wordCount += 1
num_unique_words = len(vocabDict)

# prune vocab set of words with occurences < 10, words with length < 2
one_letter_words = ['i', 'a']
wordsToDelete = []
for word, num in vocabDict.items():
    if num < 10 or (len(word) < 2 and word not in one_letter_words):
        wordsToDelete.append(word)

for word in wordsToDelete:
    del vocabDict[word]

vocabSet = set(vocabDict)
with open('vocabSet.txt', 'w') as f:
    f.writelines('%s\n' % word for word in vocabSet)

# create and dump lookup tables
word2idx = {u:i for i, u in enumerate(vocabSet)}
idx2word = np.array(list(vocabSet))
idx_not_in_vocabSet = len(word2idx)
np.save('word2idx.npy', word2idx)
np.save('idx2word.npy', idx2word)

#encode dataset, encode all words not in vocabset with same integer
with open('encoded_train_val_set.txt', 'w') as f:
    for song in train_val_set:
        for word in song:
            f.write('{} '.format(word2idx[word] if word in word2idx else idx_not_in_vocabSet))
        f.write('\n')
with open('encoded_test_set.txt', 'w') as f:
    for song in test_set:
        for word in song:
            f.write('{} '.format(word2idx[word] if word in word2idx else idx_not_in_vocabSet))
        f.write('\n')

# print some stuff
orderedWords = {k: v for k, v in sorted(vocabDict.items(), key=lambda item: item[1], reverse=True)}
topWords = {k: orderedWords[k] for k in list(orderedWords)[:50]}
bottomWords = {k: orderedWords[k] for k in list(orderedWords)[-50:]}
print('Top 50 words in word dict: ')
for key, value in topWords.items():
    print(key, value)
print('\nBottom 50 words in word dict: ')
for key, value in bottomWords.items():
    print(key, value)
print('\ntotal # of songs: ', len(songsAsList))
print('# of songs in train, val, and test sets respectively: {}, {}, {}'.format(train_len, val_len, test_len))
print('\ntotal # of words: ', wordCount)
print('# of unique words in entire set: ', num_unique_words)
print('size of vocabSet: ', len(vocabSet)+1)