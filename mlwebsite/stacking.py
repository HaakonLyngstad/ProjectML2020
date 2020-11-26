from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from training import train_model
from data_processing import get_processed_dataset_dict
from rcnn import RCNN_model
from lstm import LSTM_model
import pandas as pd
import numpy as np
import xgboost

# -- PARAMS --
MAX_NB_WORDS = 50000
MAX_SEQUENCE_LENGTH = 500

EMBEDDING_DIM_RCNN = 16
RCNN_EPOCHS = 10
RCNN_BATCH_SIZE = 128

EMBEDDING_DIM_LSTM = 16
LSTM_EPOCHS = 10
LSTM_BATCH_SIZE = 128


# gather models with optimized hyperparameters into a basemodels dictionary
def get_basemodels():
    basemodels = {}
    classifier_names = ["XGBC", "RFC", "RCNN", "LSTM"]

    classifier_list = [xgboost.XGBClassifier(max_depth=10,
                                             min_child_weight=1,
                                             scale_pos_weight=2),  # penalizes error on minority
                       RandomForestClassifier(max_depth=80, n_estimators=10),
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


# splitting the dataset in 70% Training and 30% Holdout
def split_dataset():
    processed_data, train_y, holdout_y = get_processed_dataset_dict(
        train_col="text",
        valid_col="fake",
        filename="mlwebsite/fake_job_postings_processed.csv",
        MAX_NB_WORDS=MAX_NB_WORDS,
        MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH)
    return processed_data, train_y, holdout_y


# training all the basemodels
def train_basemodels(basemodels, skf, processed_data, train_y, holdout_y):
    train_matrix = []
    holdout_matrix = []
    results_df = pd.DataFrame(columns=["Classifier", "Accuracy", "Precision", "Recall", "F1-score"])
    for clfn, clf in basemodels.items():
        print(f'Training {clfn}...')
        fold_pred = []

        #k-fold crossvalidation with i folds for each classifier in basemodels
        for i, (train_index, test_index) in enumerate(skf.split(processed_data[clfn][0], train_y)):

            # train_x and holdout_x is from the total data
            train_x, holdout_x = processed_data[clfn]

            # splitting train_x and train_y in train and test folds
            x_train_fold, x_test_fold = train_x[train_index], train_x[test_index]
            y_train_fold, y_test_fold = train_y.iloc[train_index], train_y.iloc[test_index]

            # training on the kfold train and test set
            metrics, y_test_pred = train_model(classifier=clf,
                                               name=clfn,
                                               train_x=x_train_fold,
                                               train_y=y_train_fold,
                                               valid_x=x_test_fold,
                                               valid_y=y_test_fold)

            # print results from each fold
            results_df.loc[len(results_df)] = [clfn] + metrics
            print(results_df)

            # add each predicted fold into the fold_pred array
            fold_pred.extend(y_test_pred)

        # add the predictions from the classifier to the training matrix
        train_matrix.append(fold_pred)

        # train model on all 70% of training data, predict on 30% (holdout data)
        metrics, holdout_pred = train_model(classifier=clf,
                                      name=clfn,
                                      train_x=train_x,
                                      train_y=train_y,
                                      valid_x=holdout_x,
                                      valid_y=holdout_y)
        
        results_df.loc[len(results_df)] = [clfn] + metrics
        print(results_df)
        results_df.to_csv("models/k_fold_metrics.csv", index=False)

        holdout_matrix.append(holdout_pred)
    return train_matrix, holdout_matrix


def stacking_classifier():
    basemodels = get_basemodels()
    processed_data, train_y, holdout_y = split_dataset()
    skf = StratifiedKFold(n_splits=5)
    train_matrix, holdout_matrix = train_basemodels(basemodels, skf, processed_data, train_y, holdout_y)

    # convert lists to numpy arrays
    train_matrix = np.array(train_matrix)
    holdout_matrix = np.array(holdout_matrix)

    # transpose arrays
    train_matrix = np.transpose(train_matrix)
    holdout_matrix = np.transpose(holdout_matrix)

    #metalearner = xgboost.XGBClassifier()
    # create a metalearner based on the k-fold predictions of the basemodels
    metalearner = LogisticRegression(penalty='l1', solver='saga', warm_start=True)
    metalearner.fit(train_matrix, train_y)

    # predict the metalearner on the predictions from the basemodels on the holdout data
    meta_pred = metalearner.predict(holdout_matrix)
    metrics = [accuracy_score(holdout_y, meta_pred),
               precision_score(holdout_y, meta_pred),
               recall_score(holdout_y, meta_pred),
               f1_score(holdout_y, meta_pred, pos_label=1)]

    return metrics


if __name__ == '__main__':
    stacking_classifier()
