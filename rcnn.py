from keras import (
    layers,
    models,
)
from keras.callbacks import EarlyStopping
from keras.metrics import Precision, Recall

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
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy", Precision(), Recall()])

        model.summary()

        self.model = model

    def fit(self, train_x, train_y):
        history = self.model.fit(train_x, train_y, epochs=self.EPOCH_SIZE, batch_size=self.BATCH_SIZE, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=1, min_delta=0.0001)])
        self.history = history
        return history

    def evaluate(self, valid_x, valid_y):
        results = self.model.evaluate(valid_x, valid_y, batch_size=self.BATCH_SIZE)
        return results[1:]

    def predict(self, valid_x):
        return self.model.predict(valid_x)
