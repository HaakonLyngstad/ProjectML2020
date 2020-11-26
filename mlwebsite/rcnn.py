from keras.models import Sequential
from keras import (
    layers,
    models,
)
from keras.callbacks import EarlyStopping
from keras.metrics import Precision, Recall, Accuracy


class RCNN_model:
    """
    Container class for the Keras LSTM model. Standardizes
    the usecases, so that it can be used similarily to other
    Keras models.
    """

    def __init__(self, input_length, EMBEDDING_DIM, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, EPOCH_SIZE, BATCH_SIZE):
        """
        Creates the RCNN structure. and attaches it to the class.
        """
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCH_SIZE = EPOCH_SIZE
        
        model = Sequential()
        model.add(layers.Input(input_length))
        model.add(layers.Embedding(input_dim=MAX_NB_WORDS, output_dim=EMBEDDING_DIM, input_length=input_length))
        model.add(layers.SpatialDropout1D(0.25))
        model.add(layers.Bidirectional(layers.GRU(50, return_sequences=True)))
        model.add(layers.Convolution1D(100, 3, activation="relu"))
        model.add(layers.Dense(50, activation="relu"))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(1, activation="sigmoid"))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy", Precision(), Recall()])

        model.summary()

        self.model = model

    def fit(self, train_x, train_y):
        """
        Uses the input traning data to fit the Keras model.
        """
        history = self.model.fit(train_x, train_y, epochs=self.EPOCH_SIZE, batch_size=self.BATCH_SIZE, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=1, min_delta=0.0001)])
        self.history = history
        return history

    def evaluate(self, valid_x, valid_y):
        """
        Uses the validation data to evaluate the model, 
        and returns the models metrics.
        """
        results = self.model.evaluate(valid_x, valid_y, batch_size=self.BATCH_SIZE)
        print(results)
        return results[1:] + ["NaN"]

    def predict(self, valid_x):
        return self.model.predict(valid_x)
