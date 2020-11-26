from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import naive_bayes, svm, ensemble
from sklearn.linear_model import LogisticRegression
from data_processing import get_processed_dataset_dict
from rcnn import RCNN_model
from lstm import LSTM_model
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score
from training import train_model
import pandas as pd
import numpy as np
import xgboost

# -- PARAMS --
MAX_NB_WORDS = 50000
MAX_SEQUENCE_LENGTH = 500

EMBEDDING_DIM_RCNN = 16
RCNN_EPOCHS = 2
RCNN_BATCH_SIZE = 128

EMBEDDING_DIM_LSTM = 16
LSTM_EPOCHS = 2
LSTM_BATCH_SIZE = 128


# TODO: Gather models with optimized hyperparameters into a basemodels array
def get_basemodels():
    basemodels = {}
    classifier_names = ["NB", "SVM", "RCNN", "LSTM"]

    classifier_list = [naive_bayes.MultinomialNB(),
                       svm.SVC(),
                       RCNN_model(input_length=MAX_SEQUENCE_LENGTH,
                                  EMBEDDING_DIM=EMBEDDING_DIM_RCNN,
                                  MAX_NB_WORDS=MAX_NB_WORDS,
                                  MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH,
                                  EPOCH_SIZE=RCNN_EPOCHS,
                                  BATCH_SIZE=RCNN_BATCH_SIZE),
                       LSTM_model(input_length=MAX_SEQUENCE_LENGTH,
                                  EMBEDDING_DIM=EMBEDDING_DIM_LSTM,
                                  MAX_NB_WORDS=MAX_NB_WORDS,
                                  MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH,
                                  EPOCH_SIZE=LSTM_EPOCHS,
                                  BATCH_SIZE=LSTM_BATCH_SIZE)]

    for clfn, clf in zip(classifier_names, classifier_list):
        basemodels.update({clfn: clf})
    return basemodels


# TODO: Split the original dataset into a Training and Holdout dataset.
def split_dataset():
    # splitting the dataset in 70% Training and 30% Holdout
    processed_data, train_y, holdout_y = get_processed_dataset_dict(
        train_col="text",
        valid_col="fake",
        filename="fake_job_postings_processed.csv",
        MAX_NB_WORDS=MAX_NB_WORDS,
        MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH)
    return processed_data, train_y, holdout_y

# TODO: Let Training go onwards into the upcoming loop, and save Holdout until the last part in the upcoming loop.


# TODO: Make a for loop with KFold Cross-Validation where k=4
def train_basemodels(basemodels, skf, processed_data, y):
    basemodels_result = pd.DataFrame(columns=["Classifier", "Accuracy", "Precision", "Recall"])
    clfn = list(basemodels.keys())
    clf = list(basemodels.values())
    j = 0
    full_y_pred = []
    holdout_pred = []
    for i, (train_index, test_index) in enumerate(skf.split(processed_data[clfn[j]][0], y)):
        j += 1
        train_x, holdout_x = processed_data[clfn[i]]
        x_train_fold, x_test_fold = train_x[train_index], train_x[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
        metrics, y_test_pred = train_model(classifier=clf[i],
                              name=clfn[i],
                              train_x=x_train_fold,
                              train_y=y_train_fold,
                              valid_x=x_test_fold,
                              valid_y=y_test_fold)
        basemodels_result.loc[len(basemodels_result)] = [clfn[i]] + metrics
        print(basemodels_result)
        full_y_pred.extend(y_test_pred)
        if clfn[i] in ["RCNN", "LSTM"]:
            pred = clf[i].predict(holdout_x)
            pred = np.argmax(pred, axis=1)
            holdout_pred.append(pred)
        else:
            holdout_pred.append(clf[i].predict(holdout_x))
    return basemodels_result, full_y_pred, holdout_pred


def main():
    basemodels = get_basemodels()
    processed_data, train_y, holdout_y = split_dataset()
    skf = StratifiedKFold(n_splits=len(basemodels))
    basemodels_results, full_y_pred, holdout_pred = train_basemodels(basemodels, skf, processed_data, train_y)
    # TODO: Average the holdout_pred arrays into a full_holdout_pred array.
    #print(type(full_y_pred), sum(full_y_pred), full_y_pred)
    data = np.array([holdout_pred[0], holdout_pred[1], holdout_pred[2], holdout_pred[3], holdout_y])
    #full_holdout_pred = np.average(data, axis=0)
    #full_holdout_pred = full_holdout_pred.astype(int)
    #print(full_holdout_pred, len(holdout_pred[0]), len(full_holdout_pred), sum(full_holdout_pred))
    
    # TODO: Add full_y_pred as a new feature in Training and add full_holdout_pred as a new feature in Holdout
    # TODO: Return the datasets Training and Holdout with the new features for use in the next layer.
    #train_x, holdout_x = processed_data["NB"]
    #print(train_x)
    metalearner = LogisticRegression()
    #full_y_pred = full_y_pred.reshape(-1, 1)
    #holdout_y = holdout_y.reshape(-1, 1)
    metalearner.fit(np.array(full_y_pred).reshape(-1, 1), train_y)
    predictions = metalearner.predict(np.array(holdout_pred).reshape(-1, 1))
    print(accuracy_score(holdout_y, predictions),
          precision_score(holdout_y, predictions),
          recall_score(holdout_y, predictions),
          )


if __name__ == '__main__':
    main()
