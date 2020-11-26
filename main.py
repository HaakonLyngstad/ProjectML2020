from data_processing import get_processed_dataset_dict
from training import train_model
from lstm import LSTM_model
from rcnn import RCNN_model
from sklearn import (
    naive_bayes,
    svm,
    tree,
)
from sklearn.model_selection import GridSearchCV
from sklearn import ensemble
import xgboost
import numpy as np
import pandas
import os
import shutil


train_col = "text"
valid_col = "fake"
filename = "fake_job_postings_processed.csv"

# This is fixed.
EMBEDDING_DIM_LSTM = 16
EMBEDDING_DIM_RCNN = 100

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000

# Max number of words in each text.
MAX_SEQUENCE_LENGTH = 2000

RCNN_EPOCHS = 10
RCNN_BATCH_SIZE = 32

LSTM_EPOCHS = 8
LSTM_BATCH_SIZE = 32

processed_data, train_y, valid_y = get_processed_dataset_dict(
    train_col=train_col,
    valid_col=valid_col,
    filename=filename,
    MAX_NB_WORDS=MAX_NB_WORDS,
    MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH)

input_length_rcnn = processed_data["RCNN"][0].shape[1]
input_length_lstm = processed_data["LSTM"][0].shape[1]

# AdaBoost decision tree
dct = tree.DecisionTreeClassifier(max_depth=4, max_features="auto")

# Bagging GridSearch
param_grid_BG = {'n_estimators': [5, 10],
                 'base_estimator__C': [1],
                 'base_estimator__gamma': [1]}

#param_grid_ADA = {'n_estimators': [10, 15],
#                  'learning_rate': [0.2, 0.5],
#                  'base_estimator__C': [1],
#                  'base_estimator__gamma': [1]}

param_grid_ADA = {
    'base_estimator__kernel': ["rbf", "poly", 'sigmoid', 'linear']
}


"""
param_grid_rfc = {
    'n_estimators'      : [10],
    'max_depth'         : [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'random_state'      : [0],
    #'max_features': ['auto'],
    #'criterion' :['gini']
}

                   GridSearchCV(ensemble.AdaBoostClassifier(svm.SVC(), algorithm="SAMME"),
                                param_grid=param_grid_ADA,
                                refit=True,
                                verbose=2),
"""

classifier_list = [naive_bayes.MultinomialNB(),
                   svm.SVC(),
                   ensemble.RandomForestClassifier(),

                   ensemble.AdaBoostClassifier(base_estimator=dct, n_estimators=10000, learning_rate=0.0045),
                   xgboost.XGBClassifier(),
                   #GridSearchCV(ensemble.BaggingClassifier(svm.SVC()),
                   #             param_grid=param_grid_BG,
                   #             refit=True,
                   #             verbose=2),
                   LSTM_model(input_length=input_length_lstm,
                              EMBEDDING_DIM=EMBEDDING_DIM_LSTM,
                              MAX_NB_WORDS=MAX_NB_WORDS,
                              MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH,
                              EPOCH_SIZE=LSTM_EPOCHS,
                              BATCH_SIZE=LSTM_BATCH_SIZE),
                    RCNN_model(input_length=input_length_rcnn,
                              EMBEDDING_DIM=EMBEDDING_DIM_RCNN,
                              MAX_NB_WORDS=MAX_NB_WORDS,
                              EPOCH_SIZE=RCNN_EPOCHS,
                              MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH,
                              BATCH_SIZE=RCNN_BATCH_SIZE)]

classifier_names = ["NB", "SVC", "RFC", "ADA", "XGBC", "LSTM", "RCNN"]


#classifier_list = [GridSearchCV(ensemble.RandomForestClassifier(),
                                #param_grid=param_grid_rfc,
                                #refit=True,
                                #verbose=2)]
#classifier_names = ["RFC"]


dir = 'models'
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)

results_df = pandas.DataFrame(columns=["Classifier", "Accuracy", "Precision", "Recall", "F1-Score"])
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
    print(metrics)
    results_df.loc[len(results_df)] = [clfn] + metrics
    print(results_df)
print(results_df)
results_df.to_csv("models/metrics.csv", index=False)

#print(classifier_list[3].best_params_)
#print(classifier_list[5].best_params_)

#print(classifier_list[0].best_params_)
