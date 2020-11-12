from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from keras.models import load_model
from keras.utils import to_categorical
import numpy as np
from numpy import dstack
from stacked_generalizer import StackedGeneralizer
from data_processing import get_processed_dataset_dict
import pickle


def load_all_models(classifier_names):
    all_models = list()
    for clfn in classifier_names:
        if clfn in ["RCNN", "LSTM"]:
            filename = f'models/{clfn}'
            model = load_model(filename)
        else:
            filename = f'models/{clfn}.pickle'
            model = pickle.load(open(filename, 'rb'))
        all_models.append(model)
        print(f'>loaded {filename}')
    return all_models


classifier_names = ["NB", "SVM", "RFC", "ADA", "XGBC", "BG", "RCNN", "LSTM"]
train_col = "text"
valid_col = "fake"
csvfile = "fake_job_postings_processed.csv"
MAX_NB_WORDS = 50000
MAX_SEQUENCE_LENGTH = 500

base_models = load_all_models(classifier_names)

# printing to check
print(type(base_models))
for i in range(len(base_models)):
    print(type(base_models[i]))


processed_data, train_y, valid_y = get_processed_dataset_dict(
    train_col=train_col,
    valid_col=valid_col,
    filename=csvfile,
    MAX_NB_WORDS=MAX_NB_WORDS,
    MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH)

(train_x, valid_x) = processed_data["NB"]

print(train_x[0])

# TODO: initialize base models

stacking_model = LogisticRegression()
sg = StackedGeneralizer(base_models, stacking_model, n_folds=5, verbose=True)
sg.fit(train_x, train_y)
pred = sg.predict(train_x)
pred_classes = [np.argmax(p) for p in pred]

_ = sg.evaluate(train_y, pred_classes)




"""
# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members, inputX):
    stackX = None
    for model in members:
        yhat = model.predict(inputX, verbose=0)
        # stack predictions into [rows, members, probabilities]
        if stackX is None:
            stackX = yhat
        else:
            stackX = dstack((stackX, yhat))
    # flatten predictions to [rows, members x probabilities]
    stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
    return stackX


# fit a model based on the outputs from the ensemble members
def fit_stacked_model(members, inputX, inputy):
    # create dataset using ensemble
    stackedX = stacked_dataset(members, inputX)
    # fit standalone model
    model = LogisticRegression()
    model.fit(stackedX, inputy)
    return model


# make a prediction with the stacked model
def stacked_prediction(members, model, inputX):
    # create dataset using ensemble
    stackedX = stacked_dataset(members, inputX)
    # make a prediction
    yhat = model.predict(stackedX)
    return yhat


n_members = 5
members = load_all_models(n_members)
print('Loaded %d models' % len(members))
# evaluate standalone models on test dataset
for model in members:
    testy_enc = to_categorical(ytest)
    _, acc = model.evaluate(xtest, testy_enc, verbose=0)
    print('Model Accuracy: %.3f' % acc)
# fit stacked model using the ensemble
model = fit_stacked_model(members, xtest, ytest)
# evaluate model on test set
yhat = stacked_prediction(members, model, xtest)
acc = accuracy_score(ytest, yhat)
print('Stacked Test Accuracy: %.3f' % acc)
"""