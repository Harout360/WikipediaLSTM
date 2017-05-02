from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import pandas as pd
import random
import sys

defaults = {'maxlen': 40, 'step': 3, 'units': 128, 'learnRate':0.01, 'batchSize':128, 'epochs':1, 'iterations':15}

#Units
test1 = defaults.copy()
test1['units'] = 256

#2nd LSTM Layer
test2 = defaults.copy()

#Step Size
test3 = defaults.copy()
test3['step'] = 2

#Learn rate
test4 = defaults.copy()
test4['learnRate'] = 0.001

#Batch size
test5 = defaults.copy()
test5['batchSize'] = 64

#Epochs
test6 = defaults.copy()
test6['epochs'] = 15
test6['iterations'] = 2

#Sequence length
test7 = defaults.copy()
test7['maxlen'] = 70

testBank =[defaults, test1, test2, test3, test4, test5, test6, test7]
for test in testBank:
    print(test)



filename = "Eminem/eminem-lyrics.txt"
text = open(filename, "r", encoding="utf8").read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


for testNum, test in enumerate(testBank[3:]):
    testNum += 3 # starting from where it crashed
    print("\n\nTest Number", testNum)
    # cut the text in semi-redundant sequences of maxlen characters
    maxlen = test['maxlen']
    step = test['step']
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('nb sequences:', len(sentences))

    print('Vectorization...')
    X = None
    y = None
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    print('Vectorized.')

    # build the model: a single LSTM
    print('Build model...')
    if testNum != 2:
        model = Sequential()
        model.add(LSTM(test['units'], input_shape=(maxlen, len(chars))))
    else :
        print('Adding second LSTM layer with units : ' + str(test['units']))
        model = Sequential()
        model.add(LSTM(test['units'], input_shape=(maxlen, len(chars)),return_sequences=True))
        model.add(LSTM(test['units']))
        
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=test['learnRate'])
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    print('Model built.')
    
    outputFileDir = 'Outputs/Output%d.txt' % testNum
    outputFile = open(outputFileDir, 'wb+')
    outputFile.write(('Output' + str(testNum) + '\n').encode('utf-8'))

    # train the model, output generated text after each iteration
    for iteration in range(1, test['iterations']):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        outputFile.write(('\n\nIteration' + str(iteration) + '\n').encode('utf-8'))

        model.fit(X, y,
                  batch_size=test['batchSize'],
                  epochs=test['epochs'])

        start_index = random.randint(0, len(text) - maxlen - 1)

        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print()
            print('----- diversity:', diversity)
            outputFile.write(('\n----- diversity: ' + str(diversity) + '\n').encode('utf-8'))

            generated = ''
            sentence = text[start_index: start_index + maxlen]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            outputFile.write(('----- Generating with seed: "' + sentence + '"' + '\n').encode('utf-8'))
            sys.stdout.write(generated)

            for i in range(400):
                x = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x[0, t, char_indices[char]] = 1.

                preds = model.predict(x, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                outputFile.write(next_char.encode('utf-8'))
                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()
            outputFile.write(('\n').encode('utf-8'))
    outputFile.close()
