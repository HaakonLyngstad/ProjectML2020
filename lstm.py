import numpy
import keras.layers as layers
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

# setup
input_size = 100


input_layer = layers.Input((input_size,))
