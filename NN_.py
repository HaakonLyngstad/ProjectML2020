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
from keras.metrics import Precision, Recall
import matplotlib.pyplot as plt
# df = pd.read_csv('Combined_News_DJIA.csv')

# This is fixed.
EMBEDDING_DIM = 16
# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each text.
MAX_SEQUENCE_LENGTH = 500

def build_rcnn():
    # Add an Input Layer
    input_layer = layers.Input(train_x.shape[1])

    # Add the embedding Layer
    embedding_layer = layers.Embedding(input_dim=MAX_NB_WORDS, output_dim=200, input_length=train_x.shape[1], tr)(input_layer)
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

    return model


model = create_rcnn()

epochs = 10
batch_size = 128

history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=2, min_delta=0.0001)])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

predictions = model.evaluate(valid_x, valid_y, verbose=1)
print(predictions)
