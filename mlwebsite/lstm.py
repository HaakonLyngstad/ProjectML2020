from nltk.corpus import stopwords
import re
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.layers import SpatialDropout1D
from keras.callbacks import EarlyStopping
from tokenize_text import tokenize_text
from keras.metrics import Precision, Recall
#import matplotlib.pyplot as plt

class LSTM_model:
    def __init__(self, input_length, EMBEDDING_DIM, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, EPOCH_SIZE, BATCH_SIZE):
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCH_SIZE = EPOCH_SIZE

        model = Sequential()
        model.add(Embedding(input_dim=MAX_NB_WORDS, output_dim=EMBEDDING_DIM, input_length=input_length))
        #model.add(SpatialDropout1D(0.5))
        model.add(LSTM(units=EMBEDDING_DIM, dropout=0.7, recurrent_dropout=0.7))
        model.add(Dense(units=EMBEDDING_DIM, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', Precision(), Recall()])
        model.summary()
        self.model = model

    def fit(self, train_x, train_y):
        history = self.model.fit(train_x, train_y, epochs=self.EPOCH_SIZE, batch_size=self.BATCH_SIZE, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=1, min_delta=0.0001)])
        return history

    def evaluate(self, valid_x, valid_y):
        results = self.model.evaluate(valid_x, valid_y, batch_size=self.BATCH_SIZE)
        return results[1:] + ["NaN"]

"""
model = build_lstm()
history = fit_lstm(epochs, batch_size)
results = evaluate_lstm()


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
"""