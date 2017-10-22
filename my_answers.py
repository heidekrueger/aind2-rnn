import numpy as np
from string import ascii_lowercase


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# done: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# done: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):

    model = Sequential()
    model.add(LSTM(5, input_shape = (window_size, 1)))
    model.add(Dense(1))

    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    letters = list(ascii_lowercase)
    digits = [] # we don't really want digits
    punctuation = ['!', ',', '.', ':', ';', '?']

    allowed = letters + punctuation + [' ']
    
    def cleanChar(char):
        if char in allowed:
            return char
        else:
            return ' '

    #let's simply iterate through the symbols and remove them
    #(I'm sure there's a better way)
    # We don't need to worry about creating dead space at it will be replaced later

    return ''.join(cleanChar(char) for char in text)

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):

    N = len(text)
    start_ii = range(0, N-window_size, step_size)

    inputs = [text[i : i+window_size] for i in start_ii]
    outputs = ([text[i+window_size] for i in start_ii])

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars, activation='softmax'))

    return model
