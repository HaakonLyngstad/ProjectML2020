from data_processing import get_processed_dataset_dict
from training import train_model
from lstm import LSTM_model
from rcnn import RCNN_model
from sklearn import (
    naive_bayes,
    svm
)
from sklearn import ensemble
import xgboost
import pandas
import os
import shutil

train_col = "text"
valid_col = "fake"
filename = "fake_job_postings_processed.csv"

# This is fixed.
EMBEDDING_DIM_LSTM = 200
EMBEDDING_DIM_RCNN = 200

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000

# Max number of words in each text.
MAX_SEQUENCE_LENGTH = 500

RCNN_EPOCHS = 10
RCNN_BATCH_SIZE = 128

LSTM_EPOCHS = 10
LSTM_BATCH_SIZE = 128

processed_data, train_y, valid_y = get_processed_dataset_dict(
    train_col=train_col,
    valid_col=valid_col,
    filename=filename,
    MAX_NB_WORDS=MAX_NB_WORDS,
    MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH)

input_length_rcnn = processed_data["RCNN"][0].shape[1]
input_length_lstm = processed_data["LSTM"][0].shape[1]

classifier_list = [naive_bayes.MultinomialNB(),
                   svm.SVC(),
                   ensemble.RandomForestClassifier(),
                   ensemble.AdaBoostClassifier(),
                   xgboost.XGBClassifier(),
                   ensemble.BaggingClassifier(),
                   RCNN_model(input_length=input_length_rcnn,
                              EMBEDDING_DIM=EMBEDDING_DIM_RCNN,
                              MAX_NB_WORDS=MAX_NB_WORDS,
                              MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH,
                              EPOCH_SIZE=RCNN_EPOCHS,
                              BATCH_SIZE=RCNN_BATCH_SIZE),
                   LSTM_model(input_length=input_length_lstm,
                              EMBEDDING_DIM=EMBEDDING_DIM_LSTM,
                              MAX_NB_WORDS=MAX_NB_WORDS,
                              MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH,
                              EPOCH_SIZE=LSTM_EPOCHS,
                              BATCH_SIZE=LSTM_BATCH_SIZE)]

classifier_names = ["NB", "SVM", "RFC", "ADA", "XGBC", "BG", "RCNN", "LSTM"]

dir = 'models'
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)

results_df = pandas.DataFrame(columns=["Classifier", "Accuracy", "Precision", "Recall"])
for clfl, clfn in zip(classifier_list, classifier_names):
    (train_x, valid_x) = processed_data[clfn]
    metrics, _ = train_model(
        classifier=clfl,
        name=clfn,
        train_x=train_x,
        valid_x=valid_x,
        train_y=train_y,
        valid_y=valid_y
    )
    results_df.loc[len(results_df)] = [clfn] + metrics
    print(results_df)
print(results_df)
results_df.to_csv("models/metrics.csv", index=False)
