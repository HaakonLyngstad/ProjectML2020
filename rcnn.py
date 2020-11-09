from keras import (
    layers,
    models,
)
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.layers import SpatialDropout1D
from keras.callbacks import EarlyStopping
from tokenize_text import tokenize_text
from sklearn import model_selection
from keras.metrics import Precision, Recall, Accuracy
import matplotlib.pyplot as plt
from data_processing import get_dataframe
# df = pd.read_csv('Combined_News_DJIA.csv')

# This is fixed.
EMBEDDING_DIM_RCNN = 200
# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each text.
MAX_SEQUENCE_LENGTH = 500


class RCNN_model:
    def __init__(self, input_length, EMBEDDING_DIM, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, EPOCH_SIZE, BATCH_SIZE):
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCH_SIZE = EPOCH_SIZE
        # Add an Input Layer
        input_layer = layers.Input(input_length)

        # Add the embedding Layer
        embedding_layer = layers.Embedding(input_dim=MAX_NB_WORDS, output_dim=EMBEDDING_DIM, input_length=input_length)(input_layer)
        embedding_layer = layers.SpatialDropout1D(0.65)(embedding_layer)

        # Add the recurrent layer
        rnn_layer = layers.Bidirectional(layers.GRU(50, return_sequences=True))(embedding_layer)

        # Add the convolutional Layer
        conv_layer = layers.Convolution1D(100, 3, activation="relu")(embedding_layer)

        # Add the pooling Layer
        pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

        # Add the output Layers
        output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
        output_layer1 = layers.Dropout(0.5)(output_layer1)
        output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

        # Compile the model
        model = models.Model(inputs=input_layer, outputs=output_layer2)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[Precision(), Recall()])

        model.summary()

        self.model = model

    def fit(self, train_x, train_y):
        history = self.model.fit(train_x, train_y, epochs=self.EPOCH_SIZE, batch_size=self.BATCH_SIZE, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=1, min_delta=0.0001)])
        self.history = history
        return history

    def evaluate(self, valid_x, valid_y):
        self.model.add_metric(Accuracy(), name="name")
        results = self.model.evaluate(valid_x, valid_y, batch_size=self.BATCH_SIZE)
        return results


"""
train_col = "text"
valid_col = "fake"
cols = [train_col, valid_col]
filename = "fake_job_postings_processed.csv"
df = get_dataframe(col_list=cols, filename=filename)

# Split dataset into training and validation sets
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(
    df[train_col], df[valid_col], train_size=0.7, test_size=0.3)

train_x, valid_x = tokenize_text(train_x.to_numpy().tolist(),
                                 valid_x.to_numpy().tolist(),
                                 MAX_NB_WORDS,
                                 MAX_SEQUENCE_LENGTH)

EPOCH_SIZE = 10
BATCH_SIZE = 128

rcnn = RCNN_model(input_length=train_x.shape[1], EMBEDDING_DIM=EMBEDDING_DIM_RCNN, MAX_NB_WORDS=MAX_NB_WORDS, MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH, EPOCH_SIZE=EPOCH_SIZE, BATCH_SIZE=BATCH_SIZE)

history = rcnn.fit(train_x, train_y)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

predictions = rcnn.evaluate(valid_x, valid_y)
print(predictions)
"""
