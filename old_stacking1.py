from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier
from numpy import dstack
from data_processing import get_processed_dataset_dict
import pickle


def load_all_models(classifier_names):
    all_models = list()
    for clfn in classifier_names:
        if clfn in ["RCNN", "LSTM"]:
            filename = f'models/{clfn}'
            model = load_model(filename)
            # transform to scklearn model
            model = KerasClassifier(model)
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

processed_data, train_y, valid_y = get_processed_dataset_dict(
    train_col=train_col,
    valid_col=valid_col,
    filename=csvfile,
    MAX_NB_WORDS=MAX_NB_WORDS,
    MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH)

#print(np.shape(train_y))

(train_x, valid_x) = processed_data["NB"]

# TODO: initialize base models


# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members, inputX, inputy):
    stackX = None
    for model, clfn in zip(members, classifier_names):
        print(model)
        #yhat = model.predict(inputX, verbose=0)
        #print(inputX)
        # TODO: fix this 
        if clfn in ["RCNN", "LSTM"]:
            model.fit(inputX, inputy)
        yhat = model.predict(inputX)
        print(yhat, len(yhat))
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
    stackedX = stacked_dataset(members, inputX, inputy)
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


members = load_all_models(classifier_names)
#print('Loaded %d models' % len(members))
# evaluate standalone models on test dataset

"""
for model in members:
    testy_enc = to_categorical(valid_y)
    _, acc = model.evaluate(valid_x, testy_enc, verbose=0)
    print('Model Accuracy: %.3f' % acc)
"""

# fit stacked model using the ensemble
model = fit_stacked_model(members, valid_x, valid_y)
# evaluate model on test set
yhat = stacked_prediction(members, model, valid_x)
acc = accuracy_score(valid_y, yhat)
print('Stacked Test Accuracy: %.3f' % acc)
